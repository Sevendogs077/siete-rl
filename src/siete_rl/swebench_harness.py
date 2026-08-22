"""运行 SWE-bench harness，但跳过与判分无关的全局 Docker 资源扫描。"""

from __future__ import annotations

import os
import runpy


def main() -> None:
    from docker.models.containers import ContainerCollection
    from swebench.harness import reporting

    create = ContainerCollection.create

    def create_with_limits(self, image, command=None, **kwargs):
        kwargs.update(
            mem_limit=os.environ["SIETE_HARNESS_MEMORY"],
            nano_cpus=int(float(os.environ["SIETE_HARNESS_CPUS"]) * 1_000_000_000),
            pids_limit=int(os.environ["SIETE_HARNESS_PIDS_LIMIT"]),
        )
        return create(self, image, command, **kwargs)

    ContainerCollection.create = create_with_limits

    make_run_report = reporting.make_run_report

    def make_score_report(
        predictions,
        full_dataset,
        run_id,
        _client=None,
        namespace=None,
        instance_image_tag="latest",
        env_image_tag="latest",
    ):
        return make_run_report(
            predictions,
            full_dataset,
            run_id,
            None,
            namespace,
            instance_image_tag,
            env_image_tag,
        )

    reporting.make_run_report = make_score_report
    runpy.run_module("swebench.harness.run_evaluation", run_name="__main__")


if __name__ == "__main__":
    main()
