from transformers import PretrainedConfig
from transformers import GPT2Config


class GPT2RewardConfig(PretrainedConfig):

    model_type = "gpt2-reward"

    def __init__(
        self,
        model_name="gpt2",
        max_length=512,
        backbone_config=None,
        **kwargs,
    ):

        self.model_name = model_name
        self.max_length = max_length

        # ----------------------------------------------
        # Store the GPT-2 architecture configuration.
        #
        # During export, this loads only the GPT-2
        # configuration, not the GPT-2 weights.
        # ----------------------------------------------

        if backbone_config is None:

            backbone_config = (
                GPT2Config.from_pretrained(
                    model_name
                ).to_dict()
            )

        self.backbone_config = backbone_config

        super().__init__(**kwargs)