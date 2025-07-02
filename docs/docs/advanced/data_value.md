---
title: Data Value
sidebar: 'docsSidebar'
draft: true
---

# Data Value

Or if we want to combine the two, we can do it like this:

```python {6}
flow = (
    Flow()
    .add_step("step_1", identity_handler, x=1)
    .add_step("step_2", identity_handler, x=2)
    .add_step("step_3", identity_handler)
    .set_data_value("step_3.x",  "{{step_1.return}} {{step_2.return}}") # Input: "1 2"
    .set_output("step_3_x", "step_3.x")
    .set_output("step_3_return", "step_3.return")
)
```

:::tip[Power User Feature 💪]
The template syntax (`{{...}}`) lets you combine and transform data between steps flexibly!
:::
