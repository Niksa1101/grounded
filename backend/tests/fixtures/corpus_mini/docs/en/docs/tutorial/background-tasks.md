# Background Tasks { #background-tasks }

You can define background tasks to be run *after* returning a response.

## Using `BackgroundTasks` { #using-backgroundtasks }

First, import `BackgroundTasks` and define a parameter in your *path operation function*:

{* ../../docs_src/background_tasks/tutorial001_py310.py hl[1,13] *}

## Create a task function { #create-a-task-function }

Create a function to be run as the background task:

{* ../../docs_src/background_tasks/tutorial001_py310.py ln[6:9] *}

/// tip

A task function can be a normal `def` or an `async def`.

///

### Technical Details

The class `BackgroundTasks` comes directly from `starlette.background`.
