import json

d = {'optimize': True}

with open('.LMHologram', 'w') as f:
    json.dump(d, f)
