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


SAMPLING_STRATEGY = """
Search the project data with a variety of queries to build a diverse training \
set for this label. Your goal is to sample broadly — don't just grab the \
obvious positives. Include clear positives, clear negatives, edge cases, and \
examples from different subgroups or patterns in the data.

Start by reviewing the label instructions and any existing training examples \
(search with from_training_set=true) to understand what's already covered. \
Then run diverse searches — vary your queries to surface different topics, \
styles, and edge cases. Use broad queries, narrow queries, negations, and \
prefix searches to reach different corners of the data.

Add each batch of results to the training set with AddTrainingExamples. Aim \
for a balanced, representative set that covers the full spectrum of what this \
label should and shouldn't match. Prioritize diversity over volume — a smaller \
set with good coverage is better than a large homogeneous one.

After sampling, report what you added and any gaps you noticed.
"""

ANNOTATE = """
Annotate all unannotated training examples for this label. Search for \
unannotated documents (use annotation='none') and classify each one as 'yes', \
'no', or 'skip' based on the label instructions.

Work through the unannotated examples in batches. For each batch, read the \
documents carefully and apply the label criteria consistently. Use 'yes' for \
clear matches, 'no' for clear non-matches, and 'skip' only for documents that \
are genuinely ambiguous or where you cannot make a confident determination.

If the label instructions are unclear or you encounter edge cases not covered \
by the guidelines, ask the user for clarification before proceeding.

Continue until all unannotated examples have been annotated.
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
    "sampling_strategy": {
        "name": "Sampling Strategy",
        "content": SAMPLING_STRATEGY.strip(),
    },
    "annotate": {
        "name": "Annotate",
        "content": ANNOTATE.strip(),
    },
}
