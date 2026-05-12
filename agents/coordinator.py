"""
Coordinator agent — routes incoming queries to the correct specialist agent.

Responsibilities:
- Classify query intent (document-grounded vs. general knowledge vs. tool-required).
- Emit a structured RoutingDecision with explicit reasoning so decisions are auditable.
- Delegate to RetrieverAgent or GeneralAgent accordingly.

Not yet implemented.  See schemas/routing.py for the RoutingDecision model.
"""
