"""Training candidate runtime — wraps a local model for AAT episodes.

Implements AAT's AgentRuntime interface. Forwards inference calls to a
vLLM-served model, collects prompt/response traces for RL training,
and supports LoRA adapter hot-swapping between episodes.
"""
from __future__ import annotations

import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

from src.agents.runtime.base import AgentRuntime, RuntimeResult


@dataclass
class TraceEntry:
    """A single prompt/response pair from the candidate model."""

    timestamp: float
    system_prompt: str
    user_message: str
    response: str
    turn: int
    tool_calls: List[Dict[str, Any]]
    duration_ms: float


class TrainingCandidateRuntime(AgentRuntime):
    """AAT runtime wrapping the training candidate model.

    This runtime:
    - Implements the same XML tool-calling protocol as VLLMRuntime
    - Collects prompt/response traces for RL trajectory extraction
    - Supports LoRA adapter hot-swapping via set_lora_adapter()
    - Can run in mock mode for testing without a vLLM server
    """

    def __init__(self, config: Dict[str, Any], tools: List[Any]) -> None:
        super().__init__(config, tools)
        self.endpoint: str = config.get("endpoint", "http://localhost:8000")
        self.model: str = config.get("model", "deepseek-coder-v2")
        self.max_tokens: int = config.get("max_tokens", 8192)
        self.temperature: float = config.get("temperature", 0.7)
        self.lora_adapter_path: Optional[str] = config.get("lora_adapter_path")

        # Trace collection for RL training
        self._traces: List[TraceEntry] = []

    def set_lora_adapter(self, path: Optional[str]) -> None:
        """Set or clear the LoRA adapter path for inference.

        In a real deployment, this would trigger vLLM to load/unload
        the adapter. For now it records the path for the next episode.
        """
        self.lora_adapter_path = path

    def get_traces(self) -> List[TraceEntry]:
        """Return collected prompt/response traces."""
        return list(self._traces)

    def clear_traces(self) -> None:
        """Clear collected traces (call between episodes)."""
        self._traces.clear()

    async def execute_task(
        self,
        system_prompt: str,
        user_message: str,
        max_turns: int = 20,
    ) -> RuntimeResult:
        """Execute a task with agentic tool use loop.

        Collects traces of every prompt/response for RL training.
        """
        if self._is_mock_mode():
            return self._mock_execute(system_prompt, user_message)

        enhanced_prompt = self._build_tool_prompt(system_prompt)
        conversation: List[Dict[str, str]] = [
            {"role": "system", "content": enhanced_prompt},
            {"role": "user", "content": user_message},
        ]

        all_tool_calls: List[Dict[str, Any]] = []
        files_changed: List[str] = []

        for turn in range(max_turns):
            start = time.monotonic()
            response = await self._generate(conversation)
            duration_ms = (time.monotonic() - start) * 1000

            # Parse tool calls
            tool_calls = self._parse_tool_calls(response)

            # Record trace
            current_user_msg = conversation[-1]["content"] if conversation else user_message
            self._traces.append(
                TraceEntry(
                    timestamp=time.time(),
                    system_prompt=enhanced_prompt if turn == 0 else "",
                    user_message=current_user_msg,
                    response=response,
                    turn=turn,
                    tool_calls=tool_calls,
                    duration_ms=duration_ms,
                )
            )

            if not tool_calls:
                return RuntimeResult(
                    success=True,
                    content=response,
                    turns=turn + 1,
                    tool_calls=all_tool_calls,
                    files_changed=files_changed,
                    metadata={"trace_count": len(self._traces)},
                )

            # Execute tools
            tool_results = []
            for call in tool_calls:
                result = await self._execute_tool(call["name"], call["params"])
                tool_results.append(
                    {"tool": call["name"], "params": call["params"], "result": result}
                )
                all_tool_calls.append(call)
                if hasattr(result, "files_changed") and result.files_changed:
                    files_changed.extend(result.files_changed)

            conversation.append({"role": "assistant", "content": response})
            conversation.append(
                {"role": "user", "content": self._format_tool_results(tool_results)}
            )

        return RuntimeResult(
            success=False,
            content=conversation[-2]["content"] if len(conversation) >= 2 else "",
            turns=max_turns,
            tool_calls=all_tool_calls,
            files_changed=files_changed,
            error="Maximum turns reached without task completion",
            metadata={"trace_count": len(self._traces)},
        )

    def _build_tool_prompt(self, system_prompt: str) -> str:
        """Enhance system prompt with tool documentation."""
        if not self.tools:
            return system_prompt

        tools_doc = self._format_tools_xml()
        return f"""{system_prompt}

---

# Available Tools

{tools_doc}

## How to Call Tools

To use a tool, output an XML block:

```
<tool_call>
  <name>tool_name</name>
  <arguments>
    <param_name>value</param_name>
  </arguments>
</tool_call>
```

When the task is complete, provide a summary without tool calls.
"""

    def _format_tools_xml(self) -> str:
        """Format available tools as XML documentation."""
        tools_xml = []
        for tool in self.tool_list:
            params_doc = []
            schema = tool.parameters.get("properties", {})
            required = tool.parameters.get("required", [])
            for pname, pinfo in schema.items():
                ptype = pinfo.get("type", "string")
                pdesc = pinfo.get("description", "")
                req = " (required)" if pname in required else ""
                params_doc.append(f'    <{pname} type="{ptype}"{req}>{pdesc}</{pname}>')

            pstr = "\n".join(params_doc) if params_doc else "    <no parameters needed/>"
            tools_xml.append(
                f'<tool name="{tool.name}">\n  <description>{tool.description}</description>\n'
                f"  <parameters>\n{pstr}\n  </parameters>\n</tool>"
            )
        return "\n\n".join(tools_xml)

    def _parse_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """Parse XML tool calls from model output."""
        calls: List[Dict[str, Any]] = []
        pattern = r"<tool_call>(.*?)</tool_call>"
        matches = re.findall(pattern, text, re.DOTALL)

        for match in matches:
            try:
                root = ET.fromstring(f"<tool_call>{match}</tool_call>")
                name_elem = root.find("name")
                if name_elem is None or not name_elem.text:
                    continue
                name = name_elem.text.strip()
                args_elem = root.find("arguments")
                params: Dict[str, Any] = {}
                if args_elem is not None:
                    for child in args_elem:
                        params[child.tag] = child.text or ""
                calls.append({"name": name, "params": params})
            except ET.ParseError:
                continue

        return calls

    def _format_tool_results(self, results: List[Dict[str, Any]]) -> str:
        """Format tool results for model consumption."""
        parts = []
        for item in results:
            result = item["result"]
            if result.success:
                parts.append(f"Tool: {item['tool']}\nStatus: SUCCESS\nOutput:\n{result.output}")
            else:
                error = result.error or "Unknown error"
                parts.append(f"Tool: {item['tool']}\nStatus: ERROR\nOutput:\n{error}")
        return "\n\n".join(parts)

    async def _generate(self, conversation: List[Dict[str, str]]) -> str:
        """Generate response from vLLM endpoint."""
        prompt = self._messages_to_prompt(conversation)

        request_body: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stop": ["</tool_call>", "\n\nUser:", "\n\nHuman:"],
        }

        # Include LoRA adapter if set
        if self.lora_adapter_path:
            request_body["lora_adapter"] = self.lora_adapter_path

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.endpoint}/v1/completions",
                json=request_body,
            )
            if response.status_code != 200:
                raise RuntimeError(f"vLLM returned status {response.status_code}")
            data = response.json()
            return data["choices"][0]["text"]

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert message list to prompt string."""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"\n\nUser: {content}")
            elif role == "assistant":
                parts.append(f"\n\nAssistant: {content}")
        parts.append("\n\nAssistant:")
        return "".join(parts)

    def _is_mock_mode(self) -> bool:
        """Check if running in mock mode."""
        return (
            os.environ.get("MOCK_LLM", "").lower() == "true"
            or self.endpoint.startswith("mock://")
        )

    def _mock_execute(self, system_prompt: str, user_message: str) -> RuntimeResult:
        """Mock execution for testing without vLLM."""
        mock_response = f"Analyzed task: {user_message[:100]}...\nImplementation complete."
        self._traces.append(
            TraceEntry(
                timestamp=time.time(),
                system_prompt=system_prompt,
                user_message=user_message,
                response=mock_response,
                turn=0,
                tool_calls=[],
                duration_ms=10.0,
            )
        )
        return RuntimeResult(
            success=True,
            content=mock_response,
            turns=1,
            tool_calls=[{"name": "write_file", "params": {"path": "src/example.py"}}],
            files_changed=["src/example.py"],
            metadata={"trace_count": len(self._traces)},
        )
