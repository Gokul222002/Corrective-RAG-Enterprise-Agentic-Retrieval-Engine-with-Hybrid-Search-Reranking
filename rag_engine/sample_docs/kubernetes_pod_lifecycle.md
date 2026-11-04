# Kubernetes Pod Lifecycle and Health Probing Architecture

## Pod Phases
A Kubernetes Pod follows a defined lifecycle described by its `status.phase` field. These phases represent high-level operational states:
1. **Pending**: The Pod has been accepted by the Kubernetes cluster, but one or more container images have not been created or scheduled onto a Node yet. This includes time spent downloading container images over the network.
2. **Running**: The Pod has been bound to a Node, and all containers have been created. At least one container is currently running, or is in the process of starting or restarting.
3. **Succeeded**: All containers in the Pod have terminated successfully (exit code 0) and will not be restarted.
4. **Failed**: All containers in the Pod have terminated, and at least one container has terminated in failure (non-zero exit code) or was terminated by the system.
5. **Unknown**: The state of the Pod cannot be obtained, typically due to network partition or kubelet communication failure on the host Node.

## Container Probes
The kubelet utilizes three distinct diagnostic probes to monitor container health:
- **Startup Probe**: Determines whether the application container has completed its initial initialization routines. All other probes are disabled until the startup probe succeeds. This prevents slow-starting applications from being killed prematurely.
- **Liveness Probe**: Determines whether the container needs to be restarted. If the liveness probe fails consecutively past the configured `failureThreshold`, kubelet terminates the container and triggers the container `restartPolicy`.
- **Readiness Probe**: Determines whether a container is ready to accept inbound network traffic. If a readiness probe fails, the Pod's IP address is removed from the endpoints of all matching Kubernetes Services, preventing clients from receiving 502/503 errors during overload or cache warming.

## Restart Policies
The Pod specification includes a `restartPolicy` field with possible values: `Always`, `OnFailure`, and `Never`. The default value is `Always`. Restart intervals follow an exponential back-off delay (10s, 20s, 40s...) capped at five minutes.
