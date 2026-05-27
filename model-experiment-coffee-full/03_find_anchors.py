"""
Step 3: Search the full ImageNet-1K class list for additional anchors
useful for differentiating coffee drink visual characteristics.
Looking for: cream, foam, milk, ice, chocolate, glass, whipped, cold.
"""

import torchvision.models as models

weights_meta = models.MobileNet_V2_Weights.IMAGENET1K_V1
categories = weights_meta.meta['categories']

search_terms = [
    'cream', 'foam', 'milk', 'ice', 'chocolate', 'whip',
    'glass', 'mug', 'cup', 'coffee', 'espresso', 'brew',
    'latte', 'cappuccino', 'cold', 'froth', 'drink', 'beverage',
    'eggnog', 'smoothie', 'shake', 'soda', 'cocoa',
]

print("Relevant ImageNet-1K classes:")
for i, cat in enumerate(categories):
    cat_lower = cat.lower()
    if any(term in cat_lower for term in search_terms):
        print(f"  {i:>4}: {cat}")
