# OpenHands scaffold provenance

- OpenHands SWE-Gym: `e644a2ca45c3623b27a7e6c169e3d479f0a87fbc`
  - `openhands/llm/fn_call_converter.py`: `6b3ec45ea4422e5c8067c107463c4ba331d1c86f0f68d363b819160a748d49db`
  - `openhands/agenthub/codeact_agent/function_calling.py`: `95dbfa65dedc9289a322a4ea4069d2e5c58261c93e41ab569542be824fafa57f`
  - `evaluation/utils/shared.py`: `907e0b5e3ec54b46d429ccf75c74e837d70ef6ee2826aee08354b11cf0ab183c`
- OpenHands ACI: tag `0.1.0`, commit `0698260b8e03ff2974ba81fd97ad8585a2255297`
  - editor algorithm: `3290c4a1ad6339797b3d8feeed9e95e47f25b66167c9aa6abe9e449fd4dd3d79`

Local copies are deliberately small and self-contained: `tool_protocol.py` keeps the
three function schema, XML converter rules, observation wrapper and fixed fake-user;
`openhands_editor.py` keeps the editor state machine.  Direct `Path` access from ACI
is replaced by `ContainerFileBackend`, which executes only inside the rollout
container.  The project has no runtime dependency on an OpenHands checkout.
