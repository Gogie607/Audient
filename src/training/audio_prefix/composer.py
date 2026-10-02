"""RunWeaver composer that exposes a useful handler payload."""

from __future__ import annotations

from runweaver_ml.phase_control import ObjectiveComposer, TrainingContext


class AudioObjectiveComposer(ObjectiveComposer):
    """Execute one shared forward and add independently weighted objectives."""

    def compute(self, batch, params):
        context = self.forward(TrainingContext(batch=batch), self.model)
        context.require("payload")
        total_loss = None
        for objective in self.objectives:
            objective_loss = objective.compute(context, params)
            total_loss = (
                objective_loss
                if total_loss is None
                else total_loss + objective_loss
            )
        if total_loss is None:
            raise RuntimeError("audio objective composer has no objectives")
        self.last_total_loss = float(total_loss.detach().item())
        payload = context.get("payload")
        payload.objective_losses["total"] = self.last_total_loss
        return total_loss, payload

