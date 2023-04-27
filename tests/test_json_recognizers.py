import json
from typing import Literal

import pytest

from grammar import Grammar
from recognizers import recognize
from utils.grammars import GRAMMAR_JSON
import requests

test_jsons = (
    """
            [
            {
                "_id": "64497f1d96c89b5ef6113d71",
                "index": 0,
                "guid": "6ee2ed64-492f-4d19-a63d-49dddfe6571d",
                "isActive": false,
                "balance": "$3,877.74",
                "picture": "http://placehold.it/32x32",
                "age": 25,
                "eyeColor": "blue",
                "name": "Minerva Baldwin",
                "gender": "female",
                "company": "DIGIQUE",
                "email": "minervabaldwin@digique.com",
                "phone": "+1 (989) 526-2532",
                "address": "672 River Street, Jugtown, California, 1144",
                "about": "Laboris laboris esse sit mollit magna ipsum incididunt id consectetur id. Lorem anim ullamco enim duis occaecat labore sunt qui laborum laboris id quis proident. Nisi proident nulla esse deserunt dolor consectetur anim eiusmod id proident enim eu.\r\n",
                "registered": "2017-01-24T08:03:37 +05:00",
                "latitude": 66.927314,
                "longitude": 94.102676,
                "tags": [
                    "minim",
                    "dolor",
                    "est",
                    "ea",
                    "nostrud",
                    "consectetur",
                    "velit"
                ],
                "friends": [
                    {
                        "id": 0,
                        "name": "Deloris Marks"
                    },
                    {
                        "id": 1,
                        "name": "Olivia Osborn"
                    },
                    {
                        "id": 2,
                        "name": "Finch Castaneda"
                    }
                ],
                "greeting": "Hello, Minerva Baldwin! You have 4 unread messages.",
                "favoriteFruit": "banana"
            },
            {
                "_id": "64497f1d5f83b99f40ca811d",
                "index": 1,
                "guid": "a9ea9550-5e39-4657-a06a-ec964db953c4",
                "isActive": false,
                "balance": "$1,573.17",
                "picture": "http://placehold.it/32x32",
                "age": 29,
                "eyeColor": "blue",
                "name": "Mcneil Leach",
                "gender": "male",
                "company": "ZENSUS",
                "email": "mcneilleach@zensus.com",
                "phone": "+1 (893) 482-2360",
                "address": "880 Bay Street, Singer, West Virginia, 7613",
                "about": "Excepteur non culpa cillum ipsum labore nisi consequat. Anim cillum voluptate et nulla occaecat proident. Esse et esse voluptate aute incididunt.\r\n",
                "registered": "2022-01-17T08:00:52 +05:00",
                "latitude": 23.331613,
                "longitude": -142.145775,
                "tags": [
                    "proident",
                    "est",
                    "magna",
                    "cupidatat",
                    "voluptate",
                    "est",
                    "aliqua"
                ],
                "friends": [
                    {
                        "id": 0,
                        "name": "Charlene Massey"
                    },
                    {
                        "id": 1,
                        "name": "Guthrie Odonnell"
                    },
                    {
                        "id": 2,
                        "name": "Browning Larsen"
                    }
                ],
                "greeting": "Hello, Mcneil Leach! You have 8 unread messages.",
                "favoriteFruit": "apple"
            },
            {
                "_id": "64497f1de4cc84379ccb82a7",
                "index": 2,
                "guid": "497fe1e6-8146-4a16-8df2-3785564e12e0",
                "isActive": false,
                "balance": "$2,247.39",
                "picture": "http://placehold.it/32x32",
                "age": 29,
                "eyeColor": "green",
                "name": "Jaclyn Lamb",
                "gender": "female",
                "company": "IDEGO",
                "email": "jaclynlamb@idego.com",
                "phone": "+1 (975) 454-2664",
                "address": "787 Sapphire Street, Calvary, Northern Mariana Islands, 3174",
                "about": "Commodo tempor excepteur sunt velit deserunt ipsum officia nostrud. Dolore culpa commodo est in qui deserunt pariatur est labore deserunt commodo. Labore et aliquip nisi commodo non sint minim ut.\r\n",
                "registered": "2016-06-13T04:05:31 +04:00",
                "latitude": -45.745536,
                "longitude": -67.206409,
                "tags": [
                    "laborum",
                    "culpa",
                    "do",
                    "ad",
                    "cupidatat",
                    "veniam",
                    "magna"
                ],
                "friends": [
                    {
                        "id": 0,
                        "name": "Frieda Buck"
                    },
                    {
                        "id": 1,
                        "name": "Webster Stafford"
                    },
                    {
                        "id": 2,
                        "name": "Mcclure Sweet"
                    }
                ],
                "greeting": "Hello, Jaclyn Lamb! You have 4 unread messages.",
                "favoriteFruit": "strawberry"
            },
            {
                "_id": "64497f1d22900d6251aa2799",
                "index": 3,
                "guid": "d8763b65-b760-448c-b33f-e12d19432a14",
                "isActive": true,
                "balance": "$1,523.74",
                "picture": "http://placehold.it/32x32",
                "age": 37,
                "eyeColor": "green",
                "name": "Consuelo Cohen",
                "gender": "female",
                "company": "INQUALA",
                "email": "consuelocohen@inquala.com",
                "phone": "+1 (837) 426-3923",
                "address": "604 Imlay Street, Riverton, Minnesota, 7783",
                "about": "Magna non laborum duis nostrud commodo anim fugiat minim aliquip est. Non voluptate enim laborum dolore ipsum. Exercitation laboris eu laboris ullamco tempor reprehenderit ad veniam adipisicing mollit aute. Aliquip veniam ullamco ullamco aliqua veniam pariatur nisi labore deserunt magna.\r\n",
                "registered": "2023-01-25T08:04:54 +05:00",
                "latitude": 89.919913,
                "longitude": -15.829564,
                "tags": [
                    "dolor",
                    "eiusmod",
                    "Lorem",
                    "occaecat",
                    "qui",
                    "et",
                    "velit"
                ],
                "friends": [
                    {
                        "id": 0,
                        "name": "Marcia Branch"
                    },
                    {
                        "id": 1,
                        "name": "Donovan Schmidt"
                    },
                    {
                        "id": 2,
                        "name": "Myrtle Randall"
                    }
                ],
                "greeting": "Hello, Consuelo Cohen! You have 7 unread messages.",
                "favoriteFruit": "banana"
            },
            {
                "_id": "64497f1db982e72e956a8698",
                "index": 4,
                "guid": "02e25f1b-4fb7-4bb5-b0c6-7a4e63fff38d",
                "isActive": true,
                "balance": "$1,664.68",
                "picture": "http://placehold.it/32x32",
                "age": 36,
                "eyeColor": "green",
                "name": "Mcguire Mathews",
                "gender": "male",
                "company": "BIOHAB",
                "email": "mcguiremathews@biohab.com",
                "phone": "+1 (920) 574-3637",
                "address": "278 Lloyd Street, Gloucester, Missouri, 9019",
                "about": "Magna occaecat ex aute excepteur. Ipsum laboris est magna in esse sit pariatur esse esse ad enim duis. Labore aute laboris ipsum non do. Ad Lorem cupidatat magna officia velit Lorem consequat cillum duis.\r\n",
                "registered": "2014-03-04T01:01:00 +05:00",
                "latitude": 17.583593,
                "longitude": -92.19578,
                "tags": [
                    "magna",
                    "sint",
                    "occaecat",
                    "commodo",
                    "irure",
                    "anim",
                    "minim"
                ],
                "friends": [
                    {
                        "id": 0,
                        "name": "Stacie Harper"
                    },
                    {
                        "id": 1,
                        "name": "Hart Sparks"
                    },
                    {
                        "id": 2,
                        "name": "Janice Blanchard"
                    }
                ],
                "greeting": "Hello, Mcguire Mathews! You have 10 unread messages.",
                "favoriteFruit": "banana"
            },
            {
                "_id": "64497f1dd9887cc789b49c7d",
                "index": 5,
                "guid": "5b4cfacd-28a3-4eed-88a6-2534ba548b2b",
                "isActive": false,
                "balance": "$2,407.75",
                "picture": "http://placehold.it/32x32",
                "age": 20,
                "eyeColor": "brown",
                "name": "Lynnette Austin",
                "gender": "female",
                "company": "SILODYNE",
                "email": "lynnetteaustin@silodyne.com",
                "phone": "+1 (841) 407-3870",
                "address": "188 Chauncey Street, Cutter, Colorado, 9146",
                "about": "Duis duis mollit ullamco ea non non eiusmod cupidatat dolor aliquip cupidatat. Elit in tempor in aliqua ullamco non aliqua Lorem. Velit consectetur in adipisicing reprehenderit. Non proident voluptate occaecat irure nulla voluptate aliquip sunt nisi. Ipsum veniam eu quis officia voluptate incididunt dolore sint ad tempor voluptate aliqua cupidatat nostrud.\r\n",
                "registered": "2014-03-08T12:11:14 +05:00",
                "latitude": 57.565664,
                "longitude": 54.285569,
                "tags": [
                    "sint",
                    "minim",
                    "laboris",
                    "aliqua",
                    "excepteur",
                    "minim",
                    "consectetur"
                ],
                "friends": [
                    {
                        "id": 0,
                        "name": "Meghan Wood"
                    },
                    {
                        "id": 1,
                        "name": "Meyer Watson"
                    },
                    {
                        "id": 2,
                        "name": "Shields Zimmerman"
                    }
                ],
                "greeting": "Hello, Lynnette Austin! You have 6 unread messages.",
                "favoriteFruit": "strawberry"
            }
        ]

    """,
    """
    {
        "empty_object" : {},
        "empty_array"  : [],
        "booleans"     : { "YES" : true, "NO" : false },
        "numbers"      : [ 0, 1, -2, 3.3, 4.4e5, 6.6e-7 ],
        "strings"      : [ "This", [ "And" , "That", "And a \\"b" ] ],
        "nothing"      : null
    }
""",
)


@pytest.mark.parametrize(
    "grammar, recognizer",
    [
        (Grammar.from_str(GRAMMAR_JSON, transform_regex_to_right=True), "llk"),
        (Grammar.from_str(GRAMMAR_JSON), "earley"),
        (Grammar.from_str(GRAMMAR_JSON), "lr1"),
        (Grammar.from_str(GRAMMAR_JSON), "slr"),
        (Grammar.from_str(GRAMMAR_JSON), "lr0"),
    ],
)
def test_json_recognizers(
    grammar: Grammar, recognizer: Literal["llk", "earley"]
) -> None:
    for test_json in test_jsons:
        assert recognize(grammar, test_json, recognizer=recognizer)


@pytest.mark.parametrize(
    "grammar, recognizer",
    [
        (Grammar.from_str(GRAMMAR_JSON, transform_regex_to_right=True), "llk"),
        (Grammar.from_str(GRAMMAR_JSON), "earley"),
        (Grammar.from_str(GRAMMAR_JSON), "lr1"),
        (Grammar.from_str(GRAMMAR_JSON), "slr"),
        (Grammar.from_str(GRAMMAR_JSON), "lr0"),
    ],
)
def test_with_random_json(
    grammar: Grammar, recognizer: Literal["llk", "earley"]
) -> None:
    resp = requests.get("https://random-data-api.com/api/v2/users/").json()
    assert recognize(grammar, json.dumps(resp), recognizer=recognizer)
