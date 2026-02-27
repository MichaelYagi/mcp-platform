"""
Skill Loader for MCP Servers
Implements agentskills.io specification for distributed skill discovery
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import yaml

try:
    from tools.tool_control import is_tool_enabled
except ImportError:
    def is_tool_enabled(name, category=None): return True  # fallback: all enabled


class Skill:
    """Represents a single skill with metadata"""

    def __init__(self, skill_path: Path):
        self.path = skill_path
        self.name = None
        self.description = None
        self.tags = []
        self.tools = []
        self.body = ""
        self.metadata = {}

        self._load()

    def _load(self):
        """Load and parse SKILL.md file"""
        if not self.path.exists():
            raise FileNotFoundError(f"Skill file not found: {self.path}")

        content = self.path.read_text(encoding='utf-8')

        # Parse YAML frontmatter
        if not content.startswith('---'):
            raise ValueError(f"Skill missing frontmatter: {self.path}")

        parts = content.split('---', 2)
        if len(parts) < 3:
            raise ValueError(f"Invalid frontmatter format: {self.path}")

        self.metadata = yaml.safe_load(parts[1])
        self.body = parts[2].strip()

        # Extract required fields
        self.name = self.metadata.get('name')
        self.description = self.metadata.get('description', '').strip()
        self.tags = self.metadata.get('tags', [])
        self.tools = self.metadata.get('tools', [])

        if not self.name:
            raise ValueError(f"Skill missing 'name' field: {self.path}")
        if not self.description:
            raise ValueError(f"Skill missing 'description' field: {self.path}")

    def to_dict(self) -> dict:
        """Convert skill to dictionary for serialization"""
        return {
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "tools": self.tools,
            "content": self.body,
            "path": str(self.path)
        }


class SkillRegistry:
    """Registry of all skills for a server"""

    def __init__(self):
        self.skills: Dict[str, Skill] = {}
        self.logger = logging.getLogger("skill_registry")

    def register(self, skill: Skill):
        """Register a skill"""
        self.skills[skill.name] = skill
        self.logger.info(f"   ✓ Registered skill: {skill.name}")

    def get(self, skill_name: str) -> Optional[Skill]:
        """Get a skill by name"""
        return self.skills.get(skill_name)

    def list(self) -> List[dict]:
        """List all skills with metadata (without body content)"""
        return [
            {
                "name": skill.name,
                "description": skill.description,
                "tags": skill.tags,
                "tools": skill.tools
            }
            for skill in self.skills.values()
        ]

    def get_skill_content(self, skill_name: str) -> Optional[str]:
        """Get full skill content including body"""
        skill = self.get(skill_name)
        if skill:
            return json.dumps(skill.to_dict(), indent=2)
        return None


class SkillLoader:
    """Loads skills from directory structure"""

    def __init__(self, available_tools: List[str], category: Optional[str] = None):
        """
        Args:
            available_tools: List of tool names available in this server
            category:        Optional category name (e.g. "knowledge_base") used
                             to match DISABLED_TOOLS=knowledge_base:* patterns
        """
        self.available_tools = set(available_tools)
        self.category = category
        self.logger = logging.getLogger("skill_loader")

    def load_all(self, skills_dir: Path) -> SkillRegistry:
        """
        Load all skills from a directory.

        Expected structure:
            skills/
                skill_name1/
                    SKILL.md
                skill_name2/
                    SKILL.md

        Returns:
            SkillRegistry with all loaded skills
        """
        registry = SkillRegistry()

        if not skills_dir.exists():
            self.logger.warning(f"⚠️  Skills directory not found: {skills_dir}")
            return registry

        # Find all SKILL.md files
        skill_files = list(skills_dir.rglob("SKILL.md"))

        if not skill_files:
            self.logger.info(f"ℹ️  No skills found in {skills_dir}")
            return registry

        self.logger.info(f"📚 Loading skills from {skills_dir}")

        for skill_file in skill_files:
            try:
                skill = Skill(skill_file)

                # Skip skills disabled via DISABLED_TOOLS in .env
                # Checks both exact name and category:* wildcard
                if not is_tool_enabled(skill.name, self.category):
                    self.logger.info(f"🚫 Skipping disabled skill: {skill.name}")
                    continue

                # Validate that referenced tools exist
                missing_tools = set(skill.tools) - self.available_tools
                if missing_tools:
                    self.logger.warning(
                        f"⚠️  Skill '{skill.name}' references unknown tools: {missing_tools}"
                    )

                registry.register(skill)

            except Exception as e:
                self.logger.error(f"❌ Failed to load skill from {skill_file}: {e}")

        self.logger.info(f"✓ Loaded {len(registry.skills)} skill(s)")
        return registry