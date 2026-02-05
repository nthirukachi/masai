"""
Script to fix all incorrect 28% feature importance values in FAQ.md
Replaces with actual values: EVI 15.6%, canopy 15.1%, ndvi_mean 13.4%
"""

# Read the file
with open(r'C:\masai\MQ13_AI_Crop_Health_Monitoring\documentation\FAQ.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Count original occurrences
original_count = content.count('28%')
print(f"Found {original_count} occurrences of '28%'")

# Replace patterns - be careful to maintain context
replacements = [
    # Mermaid diagrams
    ('ndvi_mean: 28%<br>moisture_index: 18%', 'evi: 15.6%<br>canopy: 15.1%<br>ndvi: 13.4%'),
    ('ndvi_mean: 28%', 'evi: 15.6%'),
    ('NDVI matters most (28%)', 'EVI matters most (15.6%)'),
    ('NDVI matters 28%', 'EVI matters 15.6%'),
    ('NDVI (28%)', 'EVI (15.6%)'),
    ('NDVI 28%', 'EVI 15.6%'),
    ('28% on NDVI', '15.6% on EVI'),
    ('28% of my attention', '15.6% of my attention'),
    ('28% importance', '15.6% importance'),
    ('28% of the model', '15.6% of the model'),
    ('ndvi_mean caused 28%', 'evi caused 15.6%'),
    ('ndvi_mean gets 28%', 'evi gets 15.6%'),
    ('ndvi_mean: 0.310 / 1.107 = 0.28 (28%)', 'evi: normalized = 0.156 (15.6%)'),
    ('= 0.28 (28%)', '= 0.156 (15.6%)'),
    ('Normalize to get<br>28% importance', 'Normalize to get<br>15.6% importance'),
    ('Is 28% importance', 'Is 15.6% importance'),
    ('28% is average', '15.6% is slightly above average'),
    ('28% is HIGH', '15.6% is GOOD'),
    ('28% is HUGE', '15.6% is VERY GOOD'),
    ('NDVI at 28%', 'EVI at 15.6%'),
    ('28% = 3.6×', '15.6% = 2×'),
    ('Importance = 28%', 'Importance = 15.6%'),
    ('NDVI (28%), Moisture (18%), Canopy (14%)', 'EVI (15.6%), Canopy (15.1%), NDVI (13.4%)'),
    ('readings 28% of the time', 'readings 15.6% of the time'),
]

# Apply replacements
for old, new in replacements:
    if old in content:
        content = content.replace(old, new)
        print(f"Replaced: '{old[:50]}...' -> '{new[:50]}...'")

# Count remaining
remaining_count = content.count('28%')
print(f"\nRemaining '28%' occurrences: {remaining_count}")

# Write back
with open(r'C:\masai\MQ13_AI_Crop_Health_Monitoring\documentation\FAQ.md', 'w', encoding='utf-8') as f:
    f.write(content)

print("\nDone! Fixed feature importance values in FAQ.md")
