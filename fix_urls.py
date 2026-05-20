import re

for filename in ['index.html', 'prd-form.html']:
    with open(filename, 'r') as f:
        content = f.read()
    
    # Replace any variation of localhost:8000 with a JS const reference
    content = content.replace("'http://localhost:8000/", "apiBase + '/")
    content = content.replace('"http://localhost:8000/', "apiBase + '/")
    content = content.replace("'${window.location.origin}/", "apiBase + '/")
    content = content.replace('`${window.location.origin}/', "apiBase + '/")
    
    # Fix closing quotes on API URLs
    content = re.sub(r"apiBase \+ '(/[^']+)'", r"apiBase + '\1'", content)
    
    with open(filename, 'w') as f:
        f.write(content)
    print(f"Fixed {filename}")

