# Distributed Consensus via the Raft Protocol

## Architectural Pillars
The Raft consensus algorithm is designed to manage replicated logs across a distributed cluster of nodes while maximizing operational understandability and fault tolerance. In Raft, consensus is decomposed into three distinct sub-problems: Leader Election, Log Replication, and Safety Invariants.

## Server States and Transitions
A Raft cluster typically consists of an odd number of servers (typically 5) to tolerate failure of minor partitions. At any given moment, each server operates in one of three mutually exclusive states:
1. **Leader**: Manages all client requests, coordinates log entry sequencing, and issues periodic heartbeats to maintain cluster authority.
2. **Follower**: Passive state responding to RPCs from leaders and candidates. If a follower receives no communication over an randomized election timeout window (150ms-300ms), it transitions to Candidate.
3. **Candidate**: Initiates a new term election, votes for itself, and broadcasts `RequestVote` RPCs to peer servers. A candidate wins an election upon receiving votes from a strict quorum majority (N/2 + 1) of cluster members.

## Log Replication and Quorum Commit
When a client sends a write command, the leader appends the command to its local log as an uncommitted entry. The leader then issues `AppendEntries` RPCs to all follower nodes. Once the entry has been replicated on a majority of nodes without inconsistency, the leader marks the entry as committed, applies it to its local state machine, and returns the execution result to the client. Subsequent heartbeats notify followers of the updated `commitIndex`.

## Raft Safety Invariants
Raft guarantees several fundamental safety properties:
- **Election Safety**: At most one leader can be elected per term.
- **Leader Append-Only**: A leader never overwrites or truncates its own log entries; it only appends new ones.
- **Log Matching Property**: If two logs contain an entry with the same index and term, then the logs are identical in all entries up through the given index.
- **Leader Completeness**: If a log entry is committed in a given term, then that entry will be present in the logs of the leaders for all higher-numbered terms.
