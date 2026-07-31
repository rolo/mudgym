from collections import OrderedDict

WEATHER_CODE_TO_NAME = OrderedDict(
    {
        "F": "fair",
        "C": "cloudy",
        "O": "overcast",
        "R": "raining",
        "T": "stormy",
        "S": "sunny",
        "B": "blizzard",
    }
)

WEATHER = list(WEATHER_CODE_TO_NAME.values())
