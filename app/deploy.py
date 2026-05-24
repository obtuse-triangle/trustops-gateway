from __future__ import annotations

import logging
import os

from datetime import datetime, timezone

logger = logging.getLogger("trustopsback")

ROLLOUT_GROUP = "argoproj.io"
ROLLOUT_VERSION = "v1alpha1"
ROLLOUT_PLURAL = "rollouts"

NAMESPACE = os.getenv("TRUSTOPS_NAMESPACE", "trustops")
CONFIGMAP_NAME = os.getenv("TRUSTOPS_CONFIGMAP", "trustops-prompt-config")
ROLLOUT_NAME = os.getenv("TRUSTOPS_ROLLOUT", "trustops-gateway")


def _build_client() -> tuple:
    """Build and return (CoreV1Api, CustomObjectsApi) using in-cluster config.

    Falls back to kubeconfig for local development.
    """
    try:
        from kubernetes import client, config

        try:
            config.load_incluster_config()
        except Exception:
            config.load_kubeconfig()

        return client.CoreV1Api(), client.CustomObjectsApi()
    except ImportError:
        logger.error("kubernetes package is not installed")
        raise
    except Exception as exc:
        logger.error("Failed to init Kubernetes client: %s", exc)
        raise


def update_config_map(
    prompt: str,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    prompt_version: str | None = None,
    namespace: str | None = None,
) -> dict:
    """Update the trustops-prompt-config ConfigMap with new prompt values.

    Returns the ConfigMap metadata dict.
    """
    ns = namespace or NAMESPACE
    core_api, _ = _build_client()

    body = {
        "data": {
            "prompt-config.yaml": _build_config_yaml(
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                prompt_version=prompt_version,
            ),
        },
    }

    try:
        result = core_api.patch_namespaced_config_map(
            name=CONFIGMAP_NAME,
            namespace=ns,
            body=body,
        )
        logger.info("ConfigMap %s/%s updated", ns, CONFIGMAP_NAME)
        return result.to_dict()
    except Exception as exc:
        logger.error("Failed to update ConfigMap %s/%s: %s", ns, CONFIGMAP_NAME, exc)
        raise


def _build_config_yaml(
    prompt: str,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    prompt_version: str | None = None,
) -> str:
    lines = [f"system_prompt: |\n  {prompt.strip().replace(chr(10), chr(10) + '  ')}"]
    if temperature is not None:
        lines.append(f"temperature: {temperature}")
    if top_p is not None:
        lines.append(f"top_p: {top_p}")
    if top_k is not None:
        lines.append(f"top_k: {top_k}")
    if prompt_version:
        lines.append(f"prompt_version: {prompt_version}")

    return "\n".join(lines) + "\n"


def restart_rollout(namespace: str | None = None) -> dict:
    """Trigger a canary rollout by adding restartAt annotation.

    Argo Rollouts detects the annotation and initiates a new rollout
    with the configured canary strategy (10% → pause → 50% → pause → 100%).
    """
    ns = namespace or NAMESPACE
    _, custom_api = _build_client()

    restart_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    patch = {
        "spec": {
            "restartAt": restart_time,
        },
    }

    try:
        result = custom_api.patch_namespaced_custom_object(
            group=ROLLOUT_GROUP,
            version=ROLLOUT_VERSION,
            namespace=ns,
            plural=ROLLOUT_PLURAL,
            name=ROLLOUT_NAME,
            body=patch,
        )
        logger.info("Rollout %s/%s restarted at %s", ns, ROLLOUT_NAME, restart_time)
        return result.get("metadata", {})
    except Exception as exc:
        logger.error("Failed to restart Rollout %s/%s: %s", ns, ROLLOUT_NAME, exc)
        raise


def trigger_deploy(
    prompt: str,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    prompt_version: str | None = None,
    namespace: str | None = None,
) -> dict:
    """Full deploy flow: update ConfigMap → trigger canary rollout.

    1. Writes the new prompt config into the ConfigMap
    2. Adds restartAt annotation to the Rollout → starts canary (10% → … → 100%)

    Returns a dict with configmap metadata and rollout metadata.
    """
    ns = namespace or NAMESPACE

    configmap_meta = update_config_map(
        prompt=prompt,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        prompt_version=prompt_version,
        namespace=ns,
    )

    rollout_meta = restart_rollout(namespace=ns)

    return {
        "namespace": ns,
        "configMap": configmap_meta.get("metadata", {}),
        "rollout": rollout_meta,
        "restartAt": rollout_meta.get("restartAt"),
    }
