
# ROADMAP

**Note: This file is in constant modification and is not complete. This roadmap is intended for internal use.**

We plan to publish a comprehensive public roadmap in the future that will include community-driven features, contribution opportunities, and recognition programs for contributors. Stay tuned for updates on how you can help shape the future of Timbal.

Currently focusing on:
1. **Core v1 to v2 Migration**: Refactoring the core framework for improved modularity
2. **File System Server**: Creating a remote connection system for platform-uploaded solutions

## List of tasks to do before moving to core v2:

- Make dump async
- Remove context from dump params
- Separate File.persist from File.serialize
- Make File.persist async
- upload_file() timbal tool
- File.persist should use upload_file()
- Remove context from state savers methods
- Centralize stuff inside TimbalPlatformSaver.put()
- Modify create_model_from_argspec to accept pydantic fields (not just timbal wrapper)
- Add update_usage() to track usage via the run context
