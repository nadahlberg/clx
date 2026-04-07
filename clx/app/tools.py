from typing import Literal

from pydantic import Field
from shortuuid import ShortUUID

from clx.app.search import And, Contains, Not, Or, Query, StartsWith, build_q
from clx.llm.agent import Tool

_su = ShortUUID()


class Search(Tool):
    """Search documents in the project using a structured query. Build queries using Contains, StartsWith, Not, Or, and And nodes."""

    query: Query = Field(
        description="A structured query object. Use {type:'contains', value:'...'} for text search, {type:'startsWith', value:'...'} for prefix search, {type:'not', query:...} for negation, {type:'or', queries:[...]} for OR, {type:'and', queries:[...]} for AND."
    )
    num_results: int = Field(
        default=10, description="Number of results to return (max 100)"
    )

    def __call__(self, agent):
        project = agent.thread.label.project
        num_results = min(self.num_results, 100)
        documents = project.documents.order_by("shuffle_key")
        documents = documents.text_query(self.query.model_dump())
        results = list(documents.values_list("text", flat=True)[:num_results])

        # Generate a short search ID and store in agent state.
        search_id = _su.random(length=8)
        searches = agent.state.setdefault("searches", {})
        searches[search_id] = {
            "query": self.query.model_dump(),
            "num_results": num_results,
            "result_count": len(results),
        }

        if not results:
            return f"[search:{search_id}] No documents found."
        return f"[search:{search_id}] {len(results)} results:\n\n" + "\n---\n".join(results)


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


class AskUser(Tool):
    """Ask the user a question with proposed answer options. Use this when you need clarification or want the user to choose between options. The question and options will be presented to the user in an interactive card."""

    question: str = Field(description="The question to ask the user")
    options: list[str] = Field(description="A list of proposed answer options")

    def __call__(self, agent):
        return "This question will be presented to the user."
