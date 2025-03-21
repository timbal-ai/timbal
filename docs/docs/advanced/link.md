---
title: Links
sidebar: 'docsSidebar'
---

Want steps to connect only under certain conditions? Easy!

### Conditional Links

```python
flow.add_link(
    "step_1", 
    "step_2", 
    condition="step_1.return == 'Hello'") # ✨ Magic condition!
```