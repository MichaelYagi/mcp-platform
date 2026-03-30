"""
Agent-to-Agent Negotiation System
Enables agents to negotiate task assignments, resource sharing, and collaboration
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class NegotiationStatus(Enum):
    """Status of a negotiation"""
    PROPOSED = "proposed"
    COUNTER_OFFERED = "counter_offered"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    EXPIRED = "expired"


class NegotiationType(Enum):
    """Types of negotiations"""
    TASK_ALLOCATION = "task_allocation"  # Who does what task
    RESOURCE_SHARING = "resource_sharing"  # Share tools/data
    COLLABORATION = "collaboration"  # Work together
    PRIORITY_SWAP = "priority_swap"  # Swap task priorities
    LOAD_BALANCING = "load_balancing"  # Redistribute work


@dataclass
class NegotiationProposal:
    """Represents a negotiation proposal"""
    proposal_id: str
    negotiation_type: NegotiationType
    initiator: str
    target: str
    terms: Dict[str, Any]
    status: NegotiationStatus
    created_at: float
    expires_at: float
    counter_offers: List[Dict] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class NegotiationEngine:
    """
    Manages agent-to-agent negotiations
    """

    def __init__(self, logger):
        self.logger = logger
        self.active_negotiations: Dict[str, NegotiationProposal] = {}
        self.negotiation_history: List[NegotiationProposal] = []

        # Negotiation policies
        self.policies = {
            "max_counter_offers": 3,
            "default_timeout": 60.0,  # seconds
            "auto_accept_threshold": 0.8  # confidence score
        }

        # Statistics
        self.stats = {
            "total_proposals": 0,
            "accepted": 0,
            "rejected": 0,
            "expired": 0,
            "counter_offered": 0
        }

    def propose(self, initiator: str, target: str, negotiation_type: NegotiationType,
                terms: Dict[str, Any], timeout: float = None) -> NegotiationProposal:
        """
        Initiate a negotiation

        Example terms for task_allocation:
        {
            "task_id": "task_123",
            "task_description": "Ingest 10 items",
            "offered_compensation": "I'll handle your next search task",
            "urgency": "high"
        }
        """
        import uuid

        proposal_id = f"neg_{uuid.uuid4().hex[:8]}"
        timeout = timeout or self.policies["default_timeout"]

        proposal = NegotiationProposal(
            proposal_id=proposal_id,
            negotiation_type=negotiation_type,
            initiator=initiator,
            target=target,
            terms=terms,
            status=NegotiationStatus.PROPOSED,
            created_at=time.time(),
            expires_at=time.time() + timeout
        )

        self.active_negotiations[proposal_id] = proposal
        self.stats["total_proposals"] += 1

        self.logger.info(f"🤝 Negotiation proposed: {initiator} → {target} ({negotiation_type.value})")
        return proposal

    def evaluate_proposal(self, agent_id: str, proposal: NegotiationProposal,
                          agent_state: Dict = None) -> Dict[str, Any]:
        """
        Evaluate a proposal from the agent's perspective
        Returns decision and reasoning
        """
        if proposal.target != agent_id:
            return {"action": "ignore", "reason": "Not the target"}

        # Check expiration
        if time.time() > proposal.expires_at:
            self._expire_proposal(proposal.proposal_id)
            return {"action": "ignore", "reason": "Expired"}

        agent_state = agent_state or {}

        # Evaluate based on negotiation type
        if proposal.negotiation_type == NegotiationType.TASK_ALLOCATION:
            return self._evaluate_task_allocation(proposal, agent_state)

        elif proposal.negotiation_type == NegotiationType.RESOURCE_SHARING:
            return self._evaluate_resource_sharing(proposal, agent_state)

        elif proposal.negotiation_type == NegotiationType.COLLABORATION:
            return self._evaluate_collaboration(proposal, agent_state)

        elif proposal.negotiation_type == NegotiationType.PRIORITY_SWAP:
            return self._evaluate_priority_swap(proposal, agent_state)

        elif proposal.negotiation_type == NegotiationType.LOAD_BALANCING:
            return self._evaluate_load_balancing(proposal, agent_state)

        return {"action": "reject", "reason": "Unknown negotiation type"}

    def _evaluate_task_allocation(self, proposal: NegotiationProposal,
                                  agent_state: Dict) -> Dict[str, Any]:
        """Evaluate task allocation proposal"""
        terms = proposal.terms

        # Check if agent is busy
        is_busy = agent_state.get("is_busy", False)
        queue_size = agent_state.get("queue_size", 0)

        # Simple heuristic: accept if not too busy
        if is_busy or queue_size > 5:
            return {
                "action": "counter",
                "reason": "Too busy right now",
                "counter_terms": {
                    **terms,
                    "proposed_delay": 300,  # Wait 5 minutes
                    "counter_compensation": "I need help with 2 tasks after this"
                }
            }

        # Check urgency
        urgency = terms.get("urgency", "normal")
        if urgency == "high" and queue_size < 3:
            return {
                "action": "accept",
                "reason": "High urgency and available capacity",
                "confidence": 0.9
            }

        if queue_size == 0:
            return {
                "action": "accept",
                "reason": "Idle and available",
                "confidence": 1.0
            }

        return {
            "action": "accept",
            "reason": "Moderate load, can accommodate",
            "confidence": 0.7
        }

    def _evaluate_resource_sharing(self, proposal: NegotiationProposal,
                                   agent_state: Dict) -> Dict[str, Any]:
        """Evaluate resource sharing proposal"""
        terms = proposal.terms
        requested_resource = terms.get("resource")

        # Check if agent has the resource
        available_resources = agent_state.get("resources", [])
        if requested_resource not in available_resources:
            return {
                "action": "reject",
                "reason": f"Resource '{requested_resource}' not available"
            }

        # Check if resource is currently in use
        resources_in_use = agent_state.get("resources_in_use", [])
        if requested_resource in resources_in_use:
            return {
                "action": "counter",
                "reason": "Resource currently in use",
                "counter_terms": {
                    **terms,
                    "available_at": time.time() + 60  # Available in 1 minute
                }
            }

        return {
            "action": "accept",
            "reason": "Resource available for sharing",
            "confidence": 0.85
        }

    def _evaluate_collaboration(self, proposal: NegotiationProposal,
                                agent_state: Dict) -> Dict[str, Any]:
        """Evaluate collaboration proposal"""
        terms = proposal.terms

        # Check compatibility
        required_skills = terms.get("required_skills", [])
        agent_skills = agent_state.get("skills", [])

        has_skills = all(skill in agent_skills for skill in required_skills)

        if not has_skills:
            return {
                "action": "reject",
                "reason": "Missing required skills"
            }

        # Check availability
        is_busy = agent_state.get("is_busy", False)
        if is_busy:
            return {
                "action": "counter",
                "reason": "Currently busy",
                "counter_terms": {
                    **terms,
                    "start_time": time.time() + 120  # Start in 2 minutes
                }
            }

        return {
            "action": "accept",
            "reason": "Skills match and available",
            "confidence": 0.9
        }

    def _evaluate_priority_swap(self, proposal: NegotiationProposal,
                                agent_state: Dict) -> Dict[str, Any]:
        """Evaluate priority swap proposal"""
        # Simple acceptance if reasonable
        return {
            "action": "accept",
            "reason": "Priority swap acceptable",
            "confidence": 0.8
        }

    def _evaluate_load_balancing(self, proposal: NegotiationProposal,
                                 agent_state: Dict) -> Dict[str, Any]:
        """Evaluate load balancing proposal"""
        queue_size = agent_state.get("queue_size", 0)

        if queue_size > 10:
            return {
                "action": "accept",
                "reason": "Overloaded, need help",
                "confidence": 1.0
            }

        if queue_size < 3:
            return {
                "action": "accept",
                "reason": "Can take more work",
                "confidence": 0.85
            }

        return {
            "action": "reject",
            "reason": "Currently at optimal load"
        }

    def respond_to_proposal(self, proposal_id: str, action: str,
                            response_data: Dict = None) -> bool:
        """
        Respond to a negotiation proposal
        action: "accept", "reject", "counter"
        """
        if proposal_id not in self.active_negotiations:
            self.logger.warning(f"⚠️ Proposal {proposal_id} not found")
            return False

        proposal = self.active_negotiations[proposal_id]

        if action == "accept":
            proposal.status = NegotiationStatus.ACCEPTED
            self.stats["accepted"] += 1
            self.logger.info(f"✅ Negotiation {proposal_id} accepted")
            self._finalize_negotiation(proposal_id)
            return True

        elif action == "reject":
            proposal.status = NegotiationStatus.REJECTED
            self.stats["rejected"] += 1
            self.logger.info(f"❌ Negotiation {proposal_id} rejected")
            self._finalize_negotiation(proposal_id)
            return True

        elif action == "counter":
            counter_terms = response_data.get("counter_terms", {})
            proposal.counter_offers.append({
                "terms": counter_terms,
                "timestamp": time.time()
            })
            proposal.status = NegotiationStatus.COUNTER_OFFERED
            self.stats["counter_offered"] += 1

            if len(proposal.counter_offers) >= self.policies["max_counter_offers"]:
                self.logger.warning(f"⚠️ Max counter offers reached for {proposal_id}")
                proposal.status = NegotiationStatus.REJECTED
                self._finalize_negotiation(proposal_id)
                return False

            self.logger.info(f"🔄 Counter offer made for {proposal_id}")
            return True

        return False

    def _expire_proposal(self, proposal_id: str):
        """Mark proposal as expired"""
        if proposal_id in self.active_negotiations:
            proposal = self.active_negotiations[proposal_id]
            proposal.status = NegotiationStatus.EXPIRED
            self.stats["expired"] += 1
            self._finalize_negotiation(proposal_id)

    def _finalize_negotiation(self, proposal_id: str):
        """Move negotiation from active to history"""
        if proposal_id in self.active_negotiations:
            proposal = self.active_negotiations.pop(proposal_id)
            self.negotiation_history.append(proposal)

    def check_expired_negotiations(self):
        """Clean up expired negotiations"""
        current_time = time.time()
        expired = []

        for proposal_id, proposal in self.active_negotiations.items():
            if current_time > proposal.expires_at:
                expired.append(proposal_id)

        for proposal_id in expired:
            self._expire_proposal(proposal_id)

        if expired:
            self.logger.debug(f"🧹 Expired {len(expired)} negotiations")

    def get_negotiation_status(self, proposal_id: str) -> Optional[NegotiationProposal]:
        """Get status of a negotiation"""
        return self.active_negotiations.get(proposal_id)

    def get_active_negotiations_for_agent(self, agent_id: str) -> List[NegotiationProposal]:
        """Get all active negotiations involving an agent"""
        return [
            p for p in self.active_negotiations.values()
            if p.initiator == agent_id or p.target == agent_id
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """Get negotiation statistics"""
        return {
            **self.stats,
            "active_negotiations": len(self.active_negotiations),
            "history_size": len(self.negotiation_history),
            "success_rate": self.stats["accepted"] / max(self.stats["total_proposals"], 1)
        }