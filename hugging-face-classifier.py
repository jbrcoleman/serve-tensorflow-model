from transformers import pipeline

text = """
Hawaii is no stranger to wildfires, but those of the past few days are being called some of the worst in the archipelago's history.

Their toll has been devastating, although what sparked the deadly fires is still under investigation.

Hurricane winds and dry weather, however, helped fuel the flames.

Drought or abnormally dry conditions across large parts of Hawaii - including the entire island of Maui - also played a role.

Wildfires generally need three ingredients: fuel in the form of biomass like vegetation or trees, a spark, and weather such as winds that drive the flames.

About 14 percent of the state is suffering from severe or moderate drought, according to the US Drought Monitor, while 80 percent of Hawaii is classed as abnormally dry.

Dry weather sucks moisture out of vegetation, meaning it can catch alight more easily and then spread.

Scientists have calculated that 90 percent of Hawaii is getting less rainfall than it did a century ago, with the period since 2008 particularly dry.

Maui itself was also under a red flag alert - meaning warm temperatures, very low humidities and stronger winds were expected to combine to produce an increased risk of fire danger - before the fires broke out.

Strong winds from Hurricane Dora, which passed Hawaii's coast on Tuesday, helped fan the flames even further.

Forecasters are expecting a stronger-than-usual Atlantic hurricane season due to record high sea surface temperatures this year, which are adding energy to the atmosphere.

Last month, the National Weather Service noted that brush fires had been reported in Maui and briefly closed a highway. Forecasters warned at the time that "the risk of fires during this year's dry season is elevated".

Scientists also note that some parts of the Hawaiian islands are covered with non-native grasses that are more flammable than native plants.

This, coupled with dry conditions, can cause a spark to ignite a fire that can spread quickly. 
"""

classifier = pipeline("zero-shot-classification")
result = classifier(
    text,
    candidate_labels=["sports", "politics", "climate"],
)

print(result)