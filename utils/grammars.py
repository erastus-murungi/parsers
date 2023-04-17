GRAMMAR1 = (
    {
        "(": "(",
        ")": ")",
        "+": "+",
        "*": "*",
    },
    """
        <E> -> <T> <E'>
        <E'> -> + <T> <E'> | <>
        <T> -> <F> <T'>
        <T'> -> * <F> <T'> | <>
        <F> -> ( <E> ) | integer
    """,
)

GRAMMAR2 = (
    {
        "+": "+",
        "-": "-",
        "*": "*",
        "/": "/",
        "(": "(",
        ")": ")",
        "^": "^",
    },
    """
       <program> -> <expression>
       <expression> -> <term> | <term> <add_op> <expression>
       <term> -> <factor> | <factor> <mult_op> <term>
       <factor> -> <power> | <power> ^ <factor>
       <power> -> <number> | ( <expression> )
       <number> -> <digit> | <digit> <number>
       <add_op> -> + | -
       <mult_op> -> * | /
       <digit> -> integer | float
   """,
)

GRAMMAR_AMBIGUOUS_PLUS_MINUS = (
    {
        "+": "+",
        "-": "-",
        "a": "a",
    },
    """
        <A> -> <A> + <A> | <A> - <A> | a
    """,
)

GRAMMAR_LR0 = (
    {
        "+": "+",
        ";": ";",
        "(": "(",
        ")": ")",
        "or_literal": "|",
    },
    """
            <E> -> <T>; | <T> + <E>
            <T> -> (<E>) | integer
    """,
)

GRAMMAR_REGEX = (
    {
        "+": "+",
        "^": "^",
        "?": "?",
        "*": "*",
        "?:": "?:",
        "[": "[",
        "]": "]",
        "(": "(",
        ")": ")",
        "\\w": "\\w",
    },
    """
        <Regex> -> <StartOfStringAnchor> <Expression>
        <Expression> -> <Subexpression> 
        <ExpressionAlternative> -> ( or_literal <Expression> ) | <>

        <Subexpression> -> <SubExpressionOne>
        <SubExpressionOne> -> <SubExpressionOne> <SubexpressionItem> | <SubexpressionItem>
        <SubexpressionItem> -> <Match> | <Group> | <Anchor> | <Backreference>
            
        <Group> -> ( <OptionalGroupNonCapturingModifier> <Expression> ) <OptionalQuantifier>
        <GroupNonCapturingModifier> -> ?:
        <OptionalQuantifier> -> <Quantifier> | <>
        <OptionalGroupNonCapturingModifier> -> <GroupNonCapturingModifier> | <>

        <Match> -> <MatchItem> <OptionalQuantifier>

        <MatchItem> -> <MatchAnyCharacter> | <MatchCharacterClass> | <MatchCharacter>
        <MatchAnyCharacter> -> .

        <MatchCharacterClass> -> <CharacterGroup> | <CharacterClass>

        <MatchCharacter> -> char

        <CharacterGroup> -> [ <OptionalCharacterGroupNegativeModifier> <OneOrMoreCharacterGroupItem> ]
        <OptionalCharacterGroupNegativeModifier> -> <CharacterGroupNegativeModifier>
        <OneOrMoreCharacterGroupItem> -> <CharacterGroupItem> <OneOrMoreCharacterGroupItem> | <CharacterGroupItem>

        <CharacterGroupNegativeModifier> -> ^ | <>
        <CharacterGroupItem> -> <CharacterClass> | <CharacterRange> | char

        <CharacterClass> -> <CharacterClassAnyWord> | <CharacterClassAnyWordInverted> | <CharacterClassAnyDecimalDigit> | <CharacterClassAnyDecimalDigitInverted>

        <CharacterClassAnyWord> -> \\w
        <CharacterClassAnyWordInverted> -> \\W
        <CharacterClassAnyDecimalDigit> -> \\d
        <CharacterClassAnyDecimalDigitInverted> -> \\D

        <CharacterRange> -> char - char
        <Quantifier> -> <QuantifierType> <OptionalLazyModifier>
        <QuantifierType> -> <ZeroOrMoreQuantifier> | <OneOrMoreQuantifier> | <ZeroOrOneQuantifier> | <RangeQuantifier>
        <OptionalLazyModifier> -> <LazyModifier> | <>

        <LazyModifier> -> ?

        <ZeroOrMoreQuantifier> -> *
        <OneOrMoreQuantifier> -> +
        <ZeroOrOneQuantifier> -> ?

        <RangeQuantifier> -> { <RangeQuantifierLowerBound> <OptionalUpperBound> }
        <OptionalUpperBound> -> , <RangeQuantifierUpperBound> | <>
        <RangeQuantifierLowerBound> -> integer
        <RangeQuantifierUpperBound> -> integer

        <Backreference> -> \\ integer

        <StartOfStringAnchor> -> ^ | <>

        <Anchor> -> <AnchorWordBoundary> | <AnchorNonWordBoundary> | <AnchorStartOfStringOnly> | <AnchorEndOfStringOnlyNotNewline> | <AnchorEndOfStringOnly> | <AnchorPreviousMatchEnd> | <AnchorEndOfString>

        <AnchorWordBoundary> -> \b
        <AnchorNonWordBoundary> -> \\B
        <AnchorStartOfStringOnly> -> \\A
        <AnchorEndOfStringOnlyNotNewline> -> \\z
        <AnchorEndOfStringOnly> -> \\Z
        <AnchorPreviousMatchEnd> -> \\G
        <AnchorEndOfString> -> $
    """,
)

GRAMMAR_DYCK = (
    {
        "(": "(",
        ")": ")",
    },
    """
        <S> -> ( <S> ) <S> | <>
    """,
)

GRAMMAR_0N1N = (
    {"0": "0", "1": "1"},
    """
    <S> -> 0 1
    <S> -> 0 <S> 1
    """,
)

GRAMMAR3 = (
    {},
    """
        <S> -> <NP> <VP>
        <S> -> <Aux> <NP> <VP>
        <S> -> <VP>
        
        <NP> -> <Pronoun>
        <NP> -> <ProperNoun>
        <NP> -> <Det> <Nominal>
        
        
        <VP> -> <Verb> <NP>
        <VP> -> <Verb>
        <VP> -> <Verb> <NP> <PP>
        <VP> -> <Verb> <PP>
        <VP> -> <VP> <PP>
        
        <Nominal> -> <Noun>
        <Nominal> -> <Nominal> <Noun>
        <Nominal> -> <Nominal> <PP>
        
        <PP> -> <Preposition> <NP>
        
        <Det> -> that | this | the | a | an
        <Noun> -> book | flight | meal | money | time
        <Verb> -> book | include | prefer
        <Pronoun> -> I | you | she | he | it | me | us | you | them
        <ProperNoun> -> Houston | TWA
        <Aux> -> does
        <Preposition> -> from | to | on | near | through
    """,
)
