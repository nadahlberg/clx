from typing import Literal

from pydantic import Field

from clx.llm.agent import Tool
from clx.query import parse_query


class Search(Tool):
    """Search documents in the project. Uses query string syntax: comma for AND, pipe for OR, tilde for NOT, caret for STARTSWITH, parentheses for grouping. AND binds lower than OR, so `A, B | C` means A AND (B OR C). Plain text does a contains search."""

    query: str = Field(description="Query string to search for")
    num_results: int = Field(
        default=10, description="Number of results to return (max 100)"
    )

    def __call__(self, agent):
        project = agent.thread.label.project
        num_results = min(self.num_results, 100)
        documents = project.documents.order_by("shuffle_key")
        if self.query.strip():
            documents = documents.query_string(self.query)
        results = list(
            documents.values_list("text", flat=True)[:num_results]
        )
        if not results:
            return "No documents found."
        return "\n---\n".join(results)


class UpdateLabelInstructions(Tool):
    """Update the instructions for the current label. Use 'append' to add new content to the end of existing instructions, or 'replace' to rewrite them entirely. You can see the current instructions in your system prompt."""

    mode: Literal["append", "replace"] = Field(
        description="'append' to add to existing instructions, 'replace' to overwrite"
    )
    content: str = Field(description="The instruction content to append or replace with")

    def __call__(self, agent):
        label = agent.thread.label
        if self.mode == "append":
            if label.instructions.strip():
                label.instructions = label.instructions.rstrip() + "\n\n" + self.content
            else:
                label.instructions = self.content
        else:
            label.instructions = self.content
        label.save(update_fields=["instructions", "updated_at"])
        return f"Label instructions updated ({self.mode})."


class UpdateProjectInstructions(Tool):
    """Update the project-level instructions. Use 'append' to add new content to the end of existing instructions, or 'replace' to rewrite them entirely. You can see the current instructions in your system prompt. Only use this if the project's manual_instructions flag is not set."""

    mode: Literal["append", "replace"] = Field(
        description="'append' to add to existing instructions, 'replace' to overwrite"
    )
    content: str = Field(description="The instruction content to append or replace with")

    def __call__(self, agent):
        project = agent.thread.label.project
        if project.manual_instructions:
            return "Cannot update: project instructions are set to manual-only."
        if self.mode == "append":
            if project.instructions.strip():
                project.instructions = project.instructions.rstrip() + "\n\n" + self.content
            else:
                project.instructions = self.content
        else:
            project.instructions = self.content
        project.save(update_fields=["instructions", "updated_at"])
        return f"Project instructions updated ({self.mode})."
