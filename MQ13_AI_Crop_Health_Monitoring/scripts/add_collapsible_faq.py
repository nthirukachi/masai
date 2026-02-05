"""
Script to add collapsible sections to FAQ.md
Converts ## Q headings to <details><summary> format
"""
import re

# Read the FAQ file
with open(r'C:\masai\MQ13_AI_Crop_Health_Monitoring\documentation\FAQ.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Split into lines for processing
lines = content.split('\n')

# Find all Q heading positions (main questions only, ## Q format)
q_pattern = re.compile(r'^## Q\d+:')
q_positions = []

for i, line in enumerate(lines):
    if q_pattern.match(line):
        q_positions.append(i)

print(f"Found {len(q_positions)} questions at positions: {q_positions}")

# Process each question section (from end to start to preserve line numbers)
for idx in range(len(q_positions) - 1, -1, -1):
    start_line = q_positions[idx]
    
    # End line is either next question start - 1, or end of file
    if idx < len(q_positions) - 1:
        end_line = q_positions[idx + 1] - 1
    else:
        end_line = len(lines) - 1
    
    # Get the question title
    title_line = lines[start_line]
    # Remove ## and keep the rest
    title = title_line.replace('## ', '').strip()
    
    # Create the details/summary wrapper
    # Before: ## Q1: Question here
    # After: <details>\n<summary><strong>Q1: Question here</strong></summary>\n\n(content)\n\n</details>
    
    # Replace the heading line with details/summary
    lines[start_line] = f'<details>\n<summary><strong>{title}</strong></summary>\n'
    
    # Find where to insert </details> - before the --- that separates questions or at end
    # Look for the last --- before the next question
    insert_pos = end_line
    # Work backwards to find appropriate closing position
    while insert_pos > start_line and lines[insert_pos].strip() == '':
        insert_pos -= 1
    
    # Insert closing tag
    lines.insert(insert_pos + 1, '\n</details>\n')

# Join and write back
new_content = '\n'.join(lines)

with open(r'C:\masai\MQ13_AI_Crop_Health_Monitoring\documentation\FAQ.md', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("Successfully added collapsible sections to FAQ.md")
