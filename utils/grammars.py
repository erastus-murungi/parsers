GRAMMAR1 = """
        <E> -> <T> <E0>
        <E0> -> '+' <T> <E0>
        <E0> ->
        <T> -> <F> <T0>
        <T0> -> '*' <F> <T0>
        <T0> ->
        <F> -> '(' <E> ')' | number
    """

GRAMMAR2 = """
       <program> -> <expression>
       <expression> -> <term> <expr_add>
       <expr_add> -> (<add_op> <expression>)?
       <term> -> <factor> <mult_term>
       <mult_term> -> (<mult_op> <term>)?
       <factor> -> <power> <power_factor>
       <power_factor> -> (<power> '^' <factor>)?
       <power> -> number | '(' <expression> ')'
       <add_op> -> '+' | '-'
       <mult_op> -> '*' | '/'
   """


GRAMMAR_AMBIGUOUS_PLUS_MINUS = """
        <A> -> <A> '+' <A> | <A> '-' <A> | 'a'
    """

GRAMMAR_LR0 = """
            <E> -> <T>';' | <T> '+' <E>
            <T> -> '(' <E> ')' | integer
            """

GRAMMAR_REGEX = """
        <regex> -> '^'? <expr>
        <expr> -> <sub_expr>+ ( '|' <expr> )?
        <sub_expr> -> <match> 
                | <group> 
                | <anchor> 
                | <backref>
        <group> -> '(' '?:'? <expr> ')' <quantifier>?
        <match> -> <match_item> <quantifier>?
        <match_item> -> '.' 
                | <match_char_class> 
                | <literal>
        <match_char_class> -> <char_group> 
                            | <char_class>
        <char_group> -> '[' '^'? <char_group_item>+ ']'
        <char_group_item> -> <char_class> 
                        | <char_range> 
                        | <literal>
        <char_class> -> '\\w' 
                    | '\\W' 
                    | '\\d' 
                    | '\\D' 
                    | '\\s' 
                    | '\\S'
        <char_range> -> char '-' char
        <quantifier> -> <quantifier_type> '?'?
        <quantifier_type> -> '*' | '+' | '?' | <range>
        <range> -> '{' integer (',' integer)? '}'
        <backref> -> '\\'integer
        <literal> -> char | word
        <anchor> -> '\\b' 
                    | '\\B' 
                    | '\\A' 
                    | '\\z' 
                    | '\\Z' 
                    | '\\G' 
                    | '$'
    """

GRAMMAR_DYCK = "<S> -> ('(' <S> ')' <S>)?"

GRAMMAR_0N1N = (
    {"0": "0", "1": "1"},
    """
    <S> -> 0 1
    <S> -> 0 <S> 1
    """,
)

GRAMMAR3 = """
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
        <Verb> -> 'book' | 'include' | 'prefer'
        <Pronoun> -> 'I' | 'you' | 'she' | 'he' | 'it' | 'me' | 'us' | 'you' | 'them'
        <ProperNoun> -> 'Houston' | 'TWA'
        <Aux> -> 'does'
        <Preposition> -> 'from' | 'to' | 'on' | 'near' | 'through'
    """

GRAMMAR_DECAF = """
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
        <bin_op> -> '+' | '-' | '*' | '/' | '%' | '&&' | '||' | '==' | '!=' | '<' | '>' | '<=' | '>='
        <literal> -> number | 'true' | 'false'
    """

RESERVED_DECAF = frozenset(
    (
        "void",
        "float",
        "int",
        "char",
        "return",
        "break",
        "continue",
        "if",
        "while",
        "else",
        "for",
        "import",
        "len",
    ),
)

GRAMMAR_LL1 = (
    """
        <S> -> 'a' <B> 'd'
        <B> -> 'c' <C> | 'e'
        <C> -> 'a' <B> | 'b' <B> | 
    """,
    {
        "a": "a",
        "b": "b",
        "c": "c",
        "d": "d",
        "e": "e",
    },
)

GRAMMAR_LL5 = """
        <S> -> 'b''b'<C>'d' | <B> 'c''c'
        <B> -> 'b'<B> | 'b'
        <C> -> 'c'<C> | 'c'
    """

GRAMMAR_JSON = """
        <Json> -> <Value>
        <Object> -> '{' <Pair> ( ',' <Pair> )* '}' 
                    | '{' '}'

        <Pair>  -> escaped_string ':' <Value>

        <Array> -> '[' <Value> ( ',' <Value> )* ']'
         | '[' ']'

        <Value> -> escaped_string
         | signed_number
         | <Object>
         | <Array>
         | 'true'
         | 'false'
         | 'null'
    """
