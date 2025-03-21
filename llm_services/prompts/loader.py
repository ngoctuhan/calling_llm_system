from jinja2 import Environment, FileSystemLoader
from pathlib import Path
from .helpers import cache_function

class PromptLoader:
    def __init__(self, template_dir=None):
        if template_dir is None:
            template_dir = Path(__file__).parent / "templates"
        
        self.env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            trim_blocks=True,
            lstrip_blocks=True
        )
    
    @cache_function
    def load_template(self, template_name: str) -> str:
        """Load a template by name"""
        print("Loading template:", template_name)
        return self.env.get_template(template_name)
    
    def render_prompt(self, template_name: str, **kwargs) -> str:
        """Render a prompt with given variables"""
        template = self.load_template(template_name)
        return template.render(**kwargs)