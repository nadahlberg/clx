PROJECT_UNDERSTANDING = """
Search the project data to understand what this project is about. Start with \
a broad sample (empty query, ~20 results), but if documents are short, pull \
more to get a representative picture.

Using the project name and data, form your best hypothesis about the project's \
goals and domain. Then ask the user a series of targeted questions to validate \
and deepen your understanding. Keep asking follow-ups until you have a thorough \
grasp of the project's aims, scope, and any nuances.

Once you have a solid understanding, update the project instructions with a \
detailed overview covering the project's purpose, domain, data characteristics, \
and any important context.

Note: the active label in your system prompt may be one of many. Focus your \
questions on the project as a whole, not the currently active label.
"""

LABEL_UNDERSTANDING = """
Search the project data to understand what this label should capture. Use the \
label name, project instructions, and the data itself to form a hypothesis \
about what documents should be included and excluded.

Be thorough in your search — look for clear positives, clear negatives, and \
edge cases. Try different query patterns to surface tricky examples that might \
be ambiguous.

Then ask the user targeted questions to validate your understanding of the \
label's criteria. Keep asking follow-ups until you can confidently distinguish \
what belongs under this label and what doesn't.

Once ready, update the label instructions with detailed, specific annotation \
guidelines. Include clear inclusion/exclusion criteria and address any edge \
cases you identified.
"""


prompt_registry = {
    "project_understanding": {
        "name": "Project Understanding",
        "content": PROJECT_UNDERSTANDING.strip(),
    },
    "label_understanding": {
        "name": "Label Understanding",
        "content": LABEL_UNDERSTANDING.strip(),
    },
}
