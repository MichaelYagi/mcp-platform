import os, re, ast

total = 0
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
                    for dec in node.decorator_list:
                        if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute) and dec.func.attr == 'tool':
                            total += 1
                            break
print(total)