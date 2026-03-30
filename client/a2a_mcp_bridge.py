from typing import Any, Dict
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model
from client.a2a_client import A2AClient


def make_a2a_tool(a2a_client: A2AClient, tool_def: Dict[str, Any]):
    """
    Create a LangChain-compatible StructuredTool that forwards calls to a remote A2A agent.
    This version:
      - Creates a proper Pydantic schema from the A2A tool definition
      - Handles multiple named parameters correctly
      - Avoids double-prefixing
      - Handles async execution correctly
    """

    # Use the remote tool name directly (no double prefixing)
    remote_name = tool_def["name"]
    local_name = f"a2a_{remote_name}"

    description = tool_def.get(
        "description",
        f"Remote A2A tool: {remote_name}"
    )

    # Extract input schema from the tool definition
    input_schema = tool_def.get('inputSchema', {})
    properties = input_schema.get('properties', {})
    required = input_schema.get('required', [])

    # Build Pydantic field definitions
    field_definitions = {}
    for prop_name, prop_schema in properties.items():
        field_type = Any  # Default type
        field_description = prop_schema.get('description', '')

        # Map JSON schema types to Python types
        prop_type = prop_schema.get('type', 'string')
        if prop_type == 'string':
            field_type = str
        elif prop_type == 'integer':
            field_type = int
        elif prop_type == 'number':
            field_type = float
        elif prop_type == 'boolean':
            field_type = bool
        elif prop_type == 'array':
            field_type = list
        elif prop_type == 'object':
            field_type = dict

        # Make optional if not required
        if prop_name not in required:
            # Use Optional syntax for Python 3.9 compatibility
            from typing import Optional
            field_definitions[prop_name] = (Optional[field_type], Field(default=None, description=field_description))
        else:
            field_definitions[prop_name] = (field_type, Field(description=field_description))

    # Create a dynamic Pydantic model for the input schema
    # If no properties, create a simple model with no fields
    if field_definitions:
        InputModel = create_model(
            f"{local_name}_input",
            **field_definitions
        )
    else:
        # Empty schema - create a base model with no fields
        class InputModel(BaseModel):
            pass

    # Create the tool function that calls A2A
    async def _run(**kwargs) -> Any:
        """
        Execute the A2A tool with the provided arguments.
        LangChain will pass named parameters based on the Pydantic schema.
        """
        # Forward the call to the remote A2A agent
        return await a2a_client.call(remote_name, kwargs)

    # Wrap in a LangChain StructuredTool
    return StructuredTool(
        name=local_name,
        description=description,
        args_schema=InputModel,
        func=lambda **kwargs: None,  # Sync placeholder (not used)
        coroutine=_run,  # Actual async implementation
    )