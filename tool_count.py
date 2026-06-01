import os, ast

total = 0
missing = []

for root, dirs, files in os.walk('servers'):
    for f in files:
        if f == 'server.py':
            path = os.path.join(root, f)
            src = open(path).read()
            tree = ast.parse(src)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if node.name in ('list_skills','read_skill','list_capabilities'):
                        continue
                    decs = []
                    for d in node.decorator_list:
                        if isinstance(d, ast.Call):
                            if isinstance(d.func, ast.Attribute):
                                decs.append(d.func.attr)
                            elif isinstance(d.func, ast.Name):
                                decs.append(d.func.id)
                        elif isinstance(d, ast.Attribute):
                            decs.append(d.attr)
                        elif isinstance(d, ast.Name):
                            decs.append(d.id)
                    if 'tool' in decs:
                        total += 1
                        if not all(x in decs for x in ['tool', 'check_tool_enabled', 'tool_meta']):
                            missing.append(f'  {path} -> {node.name} (decorators: {decs})')

print(f'Total tools: {total}')
print(f'Missing decorators: {len(missing)}')
for m in missing:
    print(m)