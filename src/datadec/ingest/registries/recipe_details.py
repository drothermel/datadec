from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from datadec.ingest.enums import DataRecipeName
from datadec.recipe_details import get_data_recipe_details_df, get_data_recipe_family


class RecipeDetails(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    data: DataRecipeName
    family: str

    total_tokens_billions: float
    pct_code: float
    pct_common_crawl: float
    pct_social_media: float
    mean_doc_length_tokens: int
    duplicate_rate_pct: float
    quality_filter_strength: int
    is_mixed_dataset: bool
    num_sources_mixed: int
    educational_content_score: int


class RecipeRegistry(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    details_by_recipe: dict[DataRecipeName, RecipeDetails]

    def __getitem__(self, key: DataRecipeName) -> RecipeDetails:
        return self.details_by_recipe[key]

    def __contains__(self, key: object) -> bool:
        return key in self.details_by_recipe

    def __iter__(self):
        return iter(self.details_by_recipe.values())

    def __len__(self) -> int:
        return len(self.details_by_recipe)


def load_recipe_registry(csv_path: Path) -> RecipeRegistry:
    df = get_data_recipe_details_df(csv_path)
    details_by_recipe: dict[DataRecipeName, RecipeDetails] = {}
    for row in df.to_dict(orient="records"):
        recipe_name = DataRecipeName(row["data"])
        family = get_data_recipe_family(row["data"])
        details_by_recipe[recipe_name] = RecipeDetails.model_validate(
            {**row, "family": family}
        )
    return RecipeRegistry(details_by_recipe=details_by_recipe)
