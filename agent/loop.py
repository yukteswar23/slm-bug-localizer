"""ReAct agent loop for bug localization.

Each step the model returns JSON {"thought": ..., "action": {"tool": ..., ...}}.
With constrained=True the reply is grammar-forced to that schema; otherwise the
JSON is parsed from free text and parse failures are logged as invalid steps.
The final step only allows `submit`, so every episode ends with an answer.

Context control: when the conversation grows past HISTORY_CHAR_BUDGET, the
oldest tool observations are replaced by a one-line stub.

Options (all off = configuration C1, experiment 03):
  func_format   regex grammar forces function names to "path.py::Name" (exp. 04)
  struct_answer functions are objects {"file", "name"} (replaces func_format)
  repeat_guard  an exact repeat of an earlier call is not executed; the model
                is told to choose a different action or submit
  budget_hints  each tool output ends with the turn count; in the last 5 turns
                the model is told to submit once it has a candidate
  ranked        submit requires ranked lists of >= 3 files and >= 3 functions
  graph_tools   adds find_definition, get_code_neighbors, search_similar_code,
                get_file_outline over a code index (C2; needs `index`)
  evidence_buffer  each reply carries an `evidence` field (one short fact from the
                last tool output); the collected facts are appended to the latest
                tool output as "Evidence so far", and old outputs are hidden in one
                chunk (down to half the budget) so the KV cache survives longer (C3)
"""

import json
import time

from .tools import GRAPH_TOOL_DOCS, FileTools, GraphTools, action_schema, tool_docs

OPTIONS = ("func_format", "struct_answer", "repeat_guard", "budget_hints", "ranked", "graph_tools",
           "evidence_buffer")

TOOLS = ["list_dir", "search_code", "read_file", "submit"]
GRAPH_TOOLS = ["find_definition", "get_code_neighbors", "search_similar_code", "get_file_outline"]
MAX_ISSUE_CHARS = 6000
HISTORY_CHAR_BUDGET = 22000  # ~6K tokens of conversation after the system prompt

SYSTEM = """You are a software engineer finding the location of a bug in a Python repository.
Read the GitHub issue, explore the code with the tools, and submit the files and functions that must be edited to fix the issue. Do not write a fix.

Tools:
{tools}

Reply with exactly one JSON object per turn:
{reply_format}
You have {max_steps} turns. Submit as soon as you are confident."""
REPLY = '{"thought": "<one or two sentences>", "action": {"tool": "<tool name>", ...arguments}}'
REPLY_EVIDENCE = ('{"thought": "<one or two sentences>", "evidence": "<one short fact from the last tool '
                  'output that matters for the bug: file, function, line; or empty>", '
                  '"action": {"tool": "<tool name>", ...arguments}}')


def parse_reply(text: str, tools: list = TOOLS):
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in reply")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    action = obj.get("action")
    if isinstance(action, dict) and isinstance(action.get("arguments"), dict):
        # free-text replies often nest {"tool": ..., "arguments": {...}}; same meaning
        obj["action"] = {"tool": action.get("tool"), **action["arguments"]}
    if not isinstance(obj.get("action"), dict) or obj["action"].get("tool") not in tools:
        raise ValueError("missing or unknown action.tool")
    return obj


def compress_history(messages: list, chunked: bool = False):
    """Hide the oldest tool outputs once the conversation exceeds the budget.
    chunked: hide down to half the budget at once, so the prompt prefix (and the
    KV cache) changes rarely instead of at every step."""
    total = sum(len(m["content"]) for m in messages[2:])
    if total <= HISTORY_CHAR_BUDGET:
        return
    target = HISTORY_CHAR_BUDGET // 2 if chunked else HISTORY_CHAR_BUDGET
    for m in messages[2:-2]:  # never touch the issue or the latest exchange
        if total <= target:
            break
        if m["role"] == "user" and not m["content"].startswith("[hidden"):
            n = m["content"].count("\n") + 1
            stub = f"[hidden to save space: earlier tool output, {n} lines]"
            total -= len(m["content"]) - len(stub)
            m["content"] = stub


def budget_note(i: int, max_steps: int) -> str:
    left = max_steps - (i + 1)
    note = f"\n\n[Turn {i + 2} of {max_steps}]"
    if left <= 5:
        note += f" Only {left} turns left: submit as soon as you have a candidate location."
    return note


def run_episode(llm, task, repo_dir, max_steps: int = 20, constrained: bool = True,
                options: dict = None, index=None) -> dict:
    opt = {k: bool((options or {}).get(k)) for k in OPTIONS}
    tools = GraphTools(repo_dir, index) if opt["graph_tools"] else FileTools(repo_dir)
    names = TOOLS[:-1] + GRAPH_TOOLS + TOOLS[-1:] if opt["graph_tools"] else TOOLS
    docs = tool_docs(opt["ranked"], opt["struct_answer"])
    if opt["graph_tools"]:  # graph tools listed before submit
        docs = docs.replace("submit(", GRAPH_TOOL_DOCS + "submit(", 1)
    issue = task.problem_statement[:MAX_ISSUE_CHARS]
    seen = {}  # canonical action -> step where it was first made
    messages = [
        {"role": "system", "content": SYSTEM.format(
            tools=docs, max_steps=max_steps,
            reply_format=REPLY_EVIDENCE if opt["evidence_buffer"] else REPLY)},
        {"role": "user", "content": f"Repository: {task.repo}\n\nIssue:\n{issue}"},
    ]
    steps, answer, t0 = [], None, time.time()
    evidence, prev_obs = [], None  # Evidence Buffer; (latest output message, its text without buffer)
    for i in range(max_steps):
        last = i == max_steps - 1
        if last:
            messages[-1]["content"] += "\n\nThis is your final turn: you must call submit now."
        schema = action_schema(["submit"] if last else names, opt["func_format"], opt["ranked"],
                               opt["struct_answer"], opt["evidence_buffer"]) \
            if constrained else None
        resp = llm.chat(messages, schema=schema)
        step = {"step": i, **{k: resp[k] for k in ("prompt_tokens", "completion_tokens",
                                                   "cached_tokens", "prompt_ms", "gen_ms",
                                                   "finish_reason")},
                "raw": resp["text"], "reasoning": resp["reasoning"], "valid": True, "tool": None, "error": None}
        messages.append({"role": "assistant", "content": resp["text"]})
        try:
            reply = parse_reply(resp["text"], names)
        except (ValueError, json.JSONDecodeError) as e:
            step.update(valid=False, error=f"invalid reply: {e}")
            obs = ('Your reply was not valid. Reply with one JSON object: '
                   '{"thought": "...", "action": {"tool": "...", ...}}')
        else:
            action = reply["action"]
            step.update(thought=reply.get("thought", ""), tool=action["tool"], action=action)
            ev = str(reply.get("evidence") or "").strip()
            if opt["evidence_buffer"] and ev:
                step["evidence"] = ev
                evidence.append(f"- (turn {i + 1}) {ev[:200]}")
            if action["tool"] == "submit":
                answer = {"files": action.get("files", []), "functions": action.get("functions", [])}
                steps.append(step)
                break
            key = json.dumps(action, sort_keys=True)
            if opt["repeat_guard"] and key in seen:
                step["repeat_blocked"] = True
                obs = (f"You already made this exact call at turn {seen[key] + 1}; repeating it gives "
                       "the same output. Choose a different action, or submit if you have found the location.")
            else:
                seen.setdefault(key, i)
                try:
                    obs = tools.call(action)
                except (ValueError, TypeError, OSError) as e:
                    step["error"] = f"tool error: {e}"
                    obs = f"Error: {e}"
        if opt["budget_hints"]:
            obs += budget_note(i, max_steps)
        step["observation"] = obs
        step["observation_chars"] = len(obs)
        steps.append(step)
        content = f"Tool output:\n{obs}"
        if opt["evidence_buffer"]:
            if prev_obs and not prev_obs[0]["content"].startswith("[hidden"):
                prev_obs[0]["content"] = prev_obs[1]  # the buffer lives only in the latest output
            msg = {"role": "user", "content": content}
            prev_obs = (msg, content)
            if evidence:
                msg["content"] += "\n\nEvidence so far:\n" + "\n".join(evidence)
            messages.append(msg)
        else:
            messages.append({"role": "user", "content": content})
        compress_history(messages, chunked=opt["evidence_buffer"])
    return {"instance_id": task.instance_id, "options": opt,
            "answer": answer or {"files": [], "functions": []},
            "submitted": answer is not None, "n_steps": len(steps),
            "wall_s": time.time() - t0, "steps": steps, "evidence": evidence}
