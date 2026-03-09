# BBQS Utility 'Sink'

This is a central repository for the BBQS consortium to use to consolidate code snippets and scripts that might be 'generally useful' for others.

There is no limit to the types of code that can be added, but we request them to be as modular as possible.



### Organization

Each utility is encapsulated as a [BIDS Study](https://bids-specification.readthedocs.io/en/stable/common-principles.html#study-dataset) style collection (regardless of whether your data is truly BIDS compliant).

That is, each capsule follows the structure:

```
[utility-identifier]/
├── code/
│   ├── [script1].py
│   ├── [script2].py
│   └── ...
├── data/
│   ├── [example_data1]
│   └── ...
├── CHANGES.md
└── README.md
```

All utilities should have descriptive `README.md` file that explains the purpose, usage, and any necessary instructions for running the code.
The use of in-code docstrings is also highly encouraged.
You may also include a `CHANGES.md` file to track any updates or modifications made to the code over time.

Including very small amounts of sample data to demonstrate intended usage is encouraged, but not required.
Designing and using a small test suite to validate the functionality of the code using this sample data is also encouraged, but not required.
