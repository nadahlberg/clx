import logging
import signal
import time

import click

from clx import init_django

logger = logging.getLogger("clx.autopilot")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [autopilot] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)

TASK_PROMPT_TEMPLATE = """\
[New Task: {prompt_name}]

{prompt_content}

[Once you have fully completed your task, call the CompleteTask tool.]
"""


def _resolve_prompt(task):
    """Get prompt content for a task, checking project customizations first."""
    from clx.app.models import Prompt
    from clx.app.prompts import prompt_registry

    try:
        prompt = Prompt.objects.get(project=task.project, prompt_id=task.prompt_id)
        return prompt.name, prompt.content
    except Prompt.DoesNotExist:
        pass

    entry = prompt_registry.get(task.prompt_id)
    if entry:
        return entry["name"], entry["content"]

    return task.prompt_id, f"(Unknown prompt: {task.prompt_id})"


def _get_thread_for_task(task):
    """Get (or lazily create) the autopilot thread for a task."""
    from clx.app.models import Thread

    if task.label:
        label = task.label
    else:
        label = task.project.active_label or task.project.labels.first()

    if not label:
        return None

    if not label.autopilot_thread_id:
        thread = Thread.objects.create(label=label)
        label.autopilot_thread = thread
        label.save(update_fields=["autopilot_thread", "updated_at"])

    return label.autopilot_thread


def _process_task(task, max_nudges, resume=False):
    """Run a single task through the autopilot agent."""
    from clx.app.agent import CLXAgent
    from clx.app.tools import CompleteTask

    thread = _get_thread_for_task(task)
    if not thread:
        logger.warning(f"No thread available for task {task.id}, skipping")
        return

    # Lock the thread and mark task in-progress.
    thread.autopilot_locked = True
    thread.save(update_fields=["autopilot_locked", "updated_at"])
    task.status = task.Status.IN_PROGRESS
    task.save(update_fields=["status", "updated_at"])

    try:
        # Build the agent with CompleteTask injected.
        tools = CLXAgent.default_tools + [CompleteTask]
        agent = CLXAgent(thread, tools=tools)

        # Determine the message to send.
        if resume:
            # User replied — just nudge the agent to continue.
            message = (
                "The user has responded above. Continue working on your task. "
                "If done, call CompleteTask."
            )
        else:
            prompt_name, prompt_content = _resolve_prompt(task)
            message = TASK_PROMPT_TEMPLATE.format(
                prompt_name=prompt_name,
                prompt_content=prompt_content,
            )

        logger.info(f"Running task: {task.prompt_id} (label={task.label})")
        result = agent.autopilot_run(message, max_nudges=max_nudges)

        if result == "completed":
            logger.info(f"Task completed: {task.prompt_id}")
            task.delete()
            task.project.update_tasks()
        elif result in ("awaiting_input", "max_steps"):
            logger.info(f"Task awaiting input: {task.prompt_id}")
            task.status = task.Status.AWAITING_INPUT
            task.save(update_fields=["status", "updated_at"])
    finally:
        thread.autopilot_locked = False
        thread.save(update_fields=["autopilot_locked", "updated_at"])


def _process_cycle(max_nudges):
    """Run one autopilot cycle across all enabled projects."""
    from clx.app.models import Project, Task

    projects = Project.objects.filter(autopilot_enabled=True)
    for project in projects:
        project.update_tasks()

        # Check for awaiting_input tasks where the user has replied.
        for task in project.tasks.filter(status=Task.Status.AWAITING_INPUT):
            thread = _get_thread_for_task(task)
            if not thread:
                continue
            last_msg = thread.messages.order_by("-created_at").first()
            if last_msg and last_msg.data.get("role") == "user":
                _process_task(task, max_nudges, resume=True)
                if not project.autopilot_enabled:
                    break

        # Refresh to check if autopilot is still enabled.
        project.refresh_from_db(fields=["autopilot_enabled"])
        if not project.autopilot_enabled:
            continue

        # Pick the next pending task (atomic claim).
        task = project.tasks.filter(status=Task.Status.PENDING).order_by("created_at").first()
        if task:
            updated = Task.objects.filter(
                id=task.id, status=Task.Status.PENDING
            ).update(status=Task.Status.IN_PROGRESS)
            if updated:
                task.refresh_from_db()
                _process_task(task, max_nudges)


def _cleanup():
    """Clear stale locks and reset in-progress tasks."""
    from clx.app.models import Task, Thread

    Thread.objects.filter(autopilot_locked=True).update(autopilot_locked=False)
    Task.objects.filter(status=Task.Status.IN_PROGRESS).update(
        status=Task.Status.PENDING
    )


@click.command()
@click.option("--interval", default=10, help="Seconds between polling cycles")
@click.option("--max-nudges", default=3, help="Max auto-nudges per task before pausing")
def autopilot(interval, max_nudges):
    """Run the autopilot loop, processing tasks for enabled projects."""
    init_django()

    _cleanup()
    logger.info(f"Autopilot started. Polling every {interval}s...")

    def handle_shutdown(signum, frame):
        logger.info("Shutting down, cleaning up locks...")
        _cleanup()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    while True:
        try:
            _process_cycle(max_nudges)
        except SystemExit:
            raise
        except Exception:
            logger.exception("Error in autopilot cycle")
        time.sleep(interval)
