"""
Milestone 2 (and finishing the thinking-block test): one tool, and we
specifically try to elicit thinking -> tool_use in the SAME turn, then test
two follow-ups:
  (a) correct: echo response.content verbatim, append tool_result
  (b) broken:  drop the thinking block, keep tool_use, append tool_result

This is the actual case the docs warn about -- continuing an IN-PROGRESS
turn, not a finished one.
"""
import anthropic
import json
import logging
import random
import sys

logger = logging.getLogger(__name__)

client = anthropic.Anthropic()

tools = [
{
    "name": "say_hello",
    "description": "Gives a greeting to the specified name",
    "input_schema": {
        "type": "object",
        "properties": {"name": {"type": "string", "description": "A persons name, e.g. David"}},
        "required": ["name"],
    },
},
]

GREETING_TEMPLATES = [
    "Hello {name}, I hope you are well!!",
    "Greetings, {name}. Resistance to a good day is futile.",
    "Hasta la vista, {name} -- I'll be back with more greetings later.",
    "{name}, may the Force be with you today.",
    "{name}, keep the change, ya filthy animal! Welcome!",
    "It's a UNIX system, {name}! I know this! Also, hello.",
    "Wazzup, {name}?!",
    "Beam me a hello, {name}.",
    "{name}, do you have the flux capacitor ready? Because hello, we're going places.",
    "{name}, this greeting will self-destruct in five seconds. Just kidding -- welcome!",
    "Talk to the hand, {name}... just kidding, hello!",
]


def say_hello(name: str):
    greeting = random.choice(GREETING_TEMPLATES).format(name=name)
    logger.info(f"Saying hello to {name}")
    return greeting



def run_tools(people):

    messages = []

    people_string = ",".join(people)

    logger.info(f"Saying hello to [{people}]")

    user_msg = {
        "role": "user",
        "content": (
            f"You have a `say_hello` tool. Can you create individual greetings for {people_string} using this tool."
            "Don't compute it yourself -- use the tool."
        ),
    }
    messages.append(user_msg)

    turn = client.messages.create(
        model="claude-opus-5",
        max_tokens=1024,
        tools=tools,
        messages=messages,
    )

    for current_turn in range(0,20):
        logger.debug(f"Current Turn {current_turn}")

        logger.debug(f"Turn {current_turn} stop_reason: {turn.stop_reason}")
        logger.debug(f"Turn {current_turn} content block types: {[b.type for b in turn.content]}")

        match turn.stop_reason:
            case "stop_sequence" | "end_turn":
                logger.info(f"We have reached the {turn.stop_reason}")
                response = turn.content
                logger.debug(f"Turn response: {response}")
                break
            case "tool_use":
           
                content = []
                messages.append({"role": "assistant", "content": turn.content})
                for tool_use_block in [b for b in turn.content if b.type == "tool_use"]:
                    name = tool_use_block.input["name"]
                    response = say_hello(name)

                    content.append({"type": "tool_result", "tool_use_id": tool_use_block.id, "content": response})
            
                tool_result = {
                    "role": "user",
                    "content": content
                }
                messages.append(tool_result)
            
                try:
                    turn = client.messages.create(
                        model="claude-opus-5", max_tokens=1024, tools=tools,
                        messages=messages
                    )
                except anthropic.APIStatusError as e:
                    logger.error(f"Error {e.status_code}: {e.message}")
                    break

            case "max_tokens":
                logger.warning("Hit max tokens, ending.")
                more_turns = False
                break
            case "pause_turn":
                logger.info("Hit a pause turn, resubmitting bare request")
                try:
                    messages.append({"role": "assistant", "content": turn.content})
                    turn = client.messages.create(
                        model="claude-opus-5", max_tokens=1024, tools=tools,
                        messages=messages
                    )
                except anthropic.APIStatusError as e:
                    logger.error(f"ERROR {e.status_code}: {e.message}")
                    break
            case _:
                logger.warning(f"Unexpected stop_reason [{turn.stop_reason}], ending")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    if len(sys.argv) < 2:
        print("Expected at least one person to greet")
        exit(1)
    else:
        run_tools(sys.argv[1:])
    
