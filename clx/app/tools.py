from typing import Literal

from pydantic import BaseModel, Field
from shortuuid import ShortUUID

from clx.app.search import Query
from clx.llm.agent import Tool

_su = ShortUUID()


class Search(Tool):
    """Search documents in the project using a structured query. Build queries using Contains, StartsWith, Not, Or, and And nodes. All text matching is case-insensitive."""

    query: Query = Field(
        description="A structured query object. Use {type:'contains', value:'...'} for text search, {type:'startsWith', value:'...'} for prefix search, {type:'not', query:...} for negation, {type:'or', queries:[...]} for OR, {type:'and', queries:[...]} for AND."
    )
    num_results: int = Field(
        default=10, description="Number of results to return (max 100)"
    )

    label_only: bool = Field(
        default=False,
        description="If true, only search documents already added to the current label's training set.",
    )
    annotation: str | None = Field(
        default=None,
        description="Filter by annotation value: 'yes', 'no', 'skip', 'none' (unannotated), or 'any' (has any annotation). Implies label_only.",
    )
    count_only: bool = Field(
        default=False,
        description="If true, return only the count of matching documents (num_results is ignored).",
    )

    def __call__(self, agent):
        project = agent.thread.label.project
        label_id = agent.thread.label_id
        documents = project.documents.order_by("shuffle_key")
        documents = documents.text_query(self.query.model_dump())
        if self.annotation:
            documents = documents.filter_annotation(label_id, self.annotation)
        elif self.label_only:
            documents = documents.training_examples(label_id)

        if self.count_only:
            total = documents.count()
            return f"{total} document(s) match."

        num_results = min(self.num_results, 100)
        rows = list(documents.values_list("id", "text")[:num_results])

        # Generate a short search ID and store in agent state.
        search_id = _su.random(length=8)
        doc_ids = [str(r[0]) for r in rows]
        searches = agent.state.setdefault("searches", {})
        searches[search_id] = {
            "query": self.query.model_dump(),
            "num_results": num_results,
            "label_only": self.label_only,
            "result_count": len(rows),
            "document_ids": doc_ids,
        }

        if not rows:
            return f"[search:{search_id}] No documents found."
        lines = [f"doc_id={r[0]}\n{r[1]}" for r in rows]
        return (
            f"[search:{search_id}] {len(rows)} results:\n\n"
            + "\n---\n".join(lines)
        )


class UpdateLabelInstructions(Tool):
    """Update the instructions for the current label. Use 'append' to add new content to the end of existing instructions, or 'replace' to rewrite them entirely. You can see the current instructions in your system prompt."""

    mode: Literal["append", "replace"] = Field(
        description="'append' to add to existing instructions, 'replace' to overwrite"
    )
    content: str = Field(
        description="The instruction content to append or replace with"
    )

    def __call__(self, agent):
        label = agent.thread.label
        if self.mode == "append":
            if label.instructions.strip():
                label.instructions = (
                    label.instructions.rstrip() + "\n\n" + self.content
                )
            else:
                label.instructions = self.content
        else:
            label.instructions = self.content
        label.save(update_fields=["instructions", "updated_at"])
        return f"Label instructions updated ({self.mode})."


class UpdateProjectInstructions(Tool):
    """Update the project-level instructions. Use 'append' to add new content to the end of existing instructions, or 'replace' to rewrite them entirely. You can see the current instructions in your system prompt."""

    mode: Literal["append", "replace"] = Field(
        description="'append' to add to existing instructions, 'replace' to overwrite"
    )
    content: str = Field(
        description="The instruction content to append or replace with"
    )

    def __call__(self, agent):
        project = agent.thread.label.project
        if self.mode == "append":
            if project.instructions.strip():
                project.instructions = (
                    project.instructions.rstrip() + "\n\n" + self.content
                )
            else:
                project.instructions = self.content
        else:
            project.instructions = self.content
        project.save(update_fields=["instructions", "updated_at"])
        return f"Project instructions updated ({self.mode})."


class AddTrainingExamples(Tool):
    """Add documents to the current label's training set. These become reference examples for classification. Preferred: pass search_id from a previous Search call. Alternatively, pass explicit document_ids (the short UUIDs shown as doc_id= in search results). Do not pass both."""

    search_id: str | None = Field(
        default=None,
        description="A search ID from a previous Search call (e.g. 'aBcDeFgH'). Adds all (or num_docs) documents from that search.",
    )
    num_docs: int | None = Field(
        default=None,
        description="Max number of documents to add from the search results. Only used with search_id.",
    )
    document_ids: list[str] | None = Field(
        default=None,
        description="Explicit list of document IDs (short UUIDs, NOT document text) to add.",
    )

    def __call__(self, agent):
        from clx.app.models import Document, LabelDocument

        label = agent.thread.label

        if self.search_id and self.document_ids:
            return "Error: provide search_id or document_ids, not both."
        if not self.search_id and not self.document_ids:
            return "Error: provide either search_id or document_ids."

        if self.search_id:
            searches = agent.state.get("searches", {})
            search = searches.get(self.search_id)
            if not search:
                return f"Error: search '{self.search_id}' not found in state."
            if "document_ids" not in search:
                return f"Error: search '{self.search_id}' has no stored document IDs. Please run the search again."
            doc_ids = search["document_ids"]
            if self.num_docs is not None:
                doc_ids = doc_ids[: self.num_docs]
        else:
            doc_ids = self.document_ids

        # Validate that all IDs actually exist to avoid FK violations.
        valid_ids = set(
            Document.objects.filter(
                id__in=doc_ids, project=label.project
            ).values_list("id", flat=True)
        )
        doc_ids = [did for did in doc_ids if did in valid_ids]

        if not doc_ids:
            return "Error: none of the provided document IDs are valid."

        objects = [
            LabelDocument(label=label, document_id=did) for did in doc_ids
        ]
        LabelDocument.objects.bulk_create(objects, ignore_conflicts=True)
        return f"Added {len(doc_ids)} training example(s) to label '{label.name}' (duplicates ignored)."


class AnnotationItem(BaseModel):
    document_id: str = Field(description="The document ID (short UUID from search results).")
    value: Literal["yes", "no", "skip"] = Field(description="The classification value.")


class Annotate(Tool):
    """Annotate training examples for the current label. Each annotation classifies a document as 'yes', 'no', or 'skip'. If an annotation already exists for a document it will be updated."""

    annotations: list[AnnotationItem] = Field(
        description="List of annotations to create or update."
    )

    def __call__(self, agent):
        from clx.app.models import ClassificationAnnotation, LabelDocument

        label = agent.thread.label
        doc_ids = [a.document_id for a in self.annotations]

        # Fetch all LabelDocument rows for these docs+label in one query.
        ld_map = dict(
            LabelDocument.objects.filter(
                label=label, document_id__in=doc_ids
            ).values_list("document_id", "id")
        )

        missing = [did for did in doc_ids if did not in ld_map]
        valid = [a for a in self.annotations if a.document_id in ld_map]

        # Bulk upsert annotations for valid docs.
        if valid:
            objects = [
                ClassificationAnnotation(
                    label_document_id=ld_map[a.document_id],
                    value=a.value,
                    source="agent",
                )
                for a in valid
            ]
            ClassificationAnnotation.objects.bulk_create(
                objects,
                update_conflicts=True,
                unique_fields=["label_document", "source"],
                update_fields=["value", "updated_at"],
            )

        parts = [f"Annotated {len(valid)} document(s)."]
        if missing:
            parts.append(f"Skipped {len(missing)} not in training set (add with AddTrainingExamples): {', '.join(missing)}")
        return " ".join(parts)


class AskUser(Tool):
    """Ask the user a question with proposed answer options. Use this when you need clarification or want the user to choose between options. The question and options will be presented to the user in an interactive card."""

    question: str = Field(description="The question to ask the user")
    options: list[str] = Field(description="A list of proposed answer options")

    def __call__(self, agent):
        return "This question will be presented to the user."
