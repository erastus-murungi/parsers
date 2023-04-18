GRAMMAR1 = (
    {
        "(": "(",
        ")": ")",
        "+": "+",
        "*": "*",
    },
    """
        <E> -> <T> <E0>
        <E0> -> ('+' <T> <E0>)?
        <T> -> <F> <T0>
        <T0> -> ('*' <F> <T0>)?
        <F> -> '(' <E> ')' | integer
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
       <factor> -> <power> | <power> '^' <factor>
       <power> -> <number> | '(' <expression> ')'
       <number> -> <digit> | <digit> <number>
       <add_op> -> '+' | '-'
       <mult_op> -> '*' | '/'
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
        <A> -> <A> '+' <A> | <A> '-' <A> | 'a'
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
            <E> -> <T>';' | <T> '+' <E>
            <T> -> '('<E>')' | integer
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
        <ExpressionAlternative> -> ( or_literal <Expression> )? | <>

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
        <S> -> ('(' <S> ')' <S>)?
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
    {"(": ")"},
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
        
        <Det> -> 'that' | 'this' | 'the' | 'a' | 'an'
        <Noun> -> 'book' | 'flight' | 'meal' | 'money' | 'time'
        <Verb> -> 'book' | 'include' | prefer
        <Pronoun> -> 'I' | 'you' | 'she' | 'he' | 'it' | 'me' | 'us' | 'you' | 'them'
        <ProperNoun> -> 'Houston' | 'TWA'
        <Aux> -> 'does'
        <Preposition> -> 'from' | 'to' | 'on' | 'near' | 'through'
    """,
)


GRAMMAR4 = (
    {
        "int": "int",
        "char": "char",
        "float": "float",
        "if": "if",
        "else": "else",
        "while": "while",
        "return": "return",
        "(": ")",
        ")": ")",
        "{": "{",
        "}": "}",
        "[": "[",
        "]": "]",
        ";": ";",
        "=": "=",
        ",": ",",
        "+": "+",
        "-": "-",
        "*": "*",
        "/": "/",
        "<": "<",
        ">": ">",
        "<=": "<=",
        ">=": ">=",
        "==": "==",
        "!=": "!=",
        "&&": "&&",
        "!": "!",
        "++": "++",
        "--": "--",
        "double_or": "||",
    },
    """
        <Program> ->  <Block>
    
        <Block> -> '{' <Declarations> <Statements> '}'
        
        <Declarations> -> (<Declarations> <Declaration>)?
        
        <Declaration> -> <Type> <Identifiers> ';'
        
        <Type> -> 'int'
               | 'float'
               | 'char'
        
        <Identifiers> -> <Identifiers> ',' word
                      | word
            
        <Statements> -> (<Statements> <Statement>)?
        
        <Statement> -> <Expression> ';'
                    | <Block>
                    | 'if' '(' <Expression> ')' <Statement> ('else' <Statement>)?
                    | 'while' '(' <Expression> ')' <Statement>
                    | 'return' <Expression> ';'
                
    
        <Expression> -> <Expression> '=' <ConditionalExpression>
                     | <ConditionalExpression>
    
        <ConditionalExpression> -> <LogicalOrExpression> ('?' <Expression> ':' <ConditionalExpression>)?
    
        <LogicalOrExpression> -> <LogicalAndExpression> (double_or <LogicalAndExpression> )*
    
        <LogicalAndExpression> -> <EqualityExpression> ('&&' <EqualityExpression>)*
    
        <EqualityExpression> -> <RelationalExpression> (<EqualityOperator> <RelationalExpression>)*
        
        <EqualityOperator> -> '==' | '!='
        
        <RelationalExpression> -> <AdditiveExpression> (<RelationalOp> <AdditiveExpression>)*
    
        <RelationalOp> -> '<' | '>' | '<=' | '>='
        
        <AdditiveExpression> -> <MultiplicativeExpression> (<PlusOrMinus> <MultiplicativeExpression>)*
    
        <PlusOrMinus> -> '+' | '-'
        
        <MultiplicativeExpression> -> <CastExpression> (<MulOps> <CastExpression>)*
        
        <MulOps> -> '*' | '/' | '%'
    
    
        <CastExpression> -> '(' <Type> ')' <CastExpression>
                         | <UnaryExpression>
    
        <UnaryOps> -> '+' | '-' | '!' | '~'
    
        <UnaryExpression> -> <UnaryOps> <UnaryExpression>
                          | <PostfixExpression>
        
        <PostfixExpression> -> <PrimaryExpression>
                            | <PostfixExpression> '[' <Expression> ']'
                            | <PostfixExpression> '(' (<Expression> (',' <Expression>)*)? ')'
                            | <PostfixExpression> '.' word
                            | <PostfixExpression> '->' word
                            | <PostfixExpression> '++'
                            | <PostfixExpression> '--'
        
        <PrimaryExpression> -> word
                             | integer
                             | float
                             | char
                             | word
                             | '(' <Expression> ')'
    """,
)


DECAF_GRAMMAR = (
    {
        "int": "int",
        "char": "char",
        "float": "float",
        "if": "if",
        "else": "else",
        "while": "while",
        "return": "return",
        "break": "break",
        "continue": "continue",
        "(": "(",
        ")": ")",
        "{": "{",
        "}": "}",
        "[": "[",
        "]": "]",
        ";": ";",
        "=": "=",
        ",": ",",
        "+": "+",
        "-": "-",
        "*": "*",
        "/": "/",
        "<": "<",
        ">": ">",
        "<=": "<=",
        ">=": ">=",
        "==": "==",
        "!=": "!=",
        "&&": "&&",
        "!": "!",
        "++": "++",
        "--": "--",
        "*=": "*=",
        "/=": "/=",
        "%=": "%=",
        "+=": "+=",
        "-=": "-=",
        "<<=": "<<=",
        "double_or": "||",
    },
    """
        <program> ->  <import_decl>* <field_decl>* <method_decl>*
        <import_decl> -> 'import' word (',' word)* ';'
        <field_decl> -> <type> <var_decl> (',' <var_decl>)* ';'
        <var_decl> -> word | word '[' integer ']'
        <method_decl> -> <method_return> word '(' <method_params> (<type> word)? ')' <block>
        <method_params> -> (<type> word ',' )*
        <method_return> -> <type> | 'void' 
        <block> -> '{' <field_decl>* <statement>* '}'
        <type> -> 'int' | 'float' | 'char' | 'boolean'
        <statement> -> <location> <assign_expr> ';'
                    | <method_call> ';'
                    | 'if' '(' <expr> ')' <block> ('else' <block>)?
                    | 'while' '(' <expr> ')' <block>
                    | 'return' <expr>? ';'
                    | 'for' '(' <for_init>? ';' <expr>? ';' <for_update>? ')' <block>
                    | 'break' ';'
                    | 'continue' ';'
        <for_init> -> <type>? word '=' <expr>
        <for_update> -> <location> <update>
        <update> -> <increment> | <compound_assign_op> <expr>
        <increment> -> '++' | '--'
        <compound_assign_op> -> '+=' | '-=' | '*=' | '/='
        <assign_op> -> '=' | <compound_assign_op>
        <assign_expr> -> <assign_op> <expr> | <increment>
        <method_call> -> word '(' (<expr> (',' <expr>)*)? ')'
        <location> -> word  | word '[' <expr> ']'
        <expr> -> <location> 
                | <method_call> 
                | <literal> 
                | <expr> <bin_op> <expr> 
                | '!' <expr> 
                | '-' <expr> 
                | '(' <expr> ')'
                | <expr> '?' <expr> ':' <expr>
                | 'len' '(' word ')'
        <bin_op> -> '+' | '-' | '*' | '/' | '%' | '&&' | double_or | '==' | '!=' | '<' | '>' | '<=' | '>='
        <literal> -> integer | float | char | 'true' | 'false'
    """,
)
