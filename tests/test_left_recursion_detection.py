import pytest

from grammar import Grammar
from grammar.left_recursion import compute_left_recursion_non_terminals


@pytest.mark.parametrize(
    "grammar_str,expected",
    [
        (
            """
    <A> -> <B> 'r'
    <B> -> <C> 'd'
    <C> -> <A> 't'
    """,
            {"A", "B", "C"},
        ),
        (
            """
    <S> -> <S> 'a'
    <S> ->
    """,
            {"S"},
        ),
        (
            """
    <S> -> <B> <S>
    <B> -> 'b' |  
    """,
            {"S"},
        ),
        (
            """
    <A> -> <A> <B> 'd'
    <A> -> <A> 'a'
    <A> -> 'a'
    <B> -> <B> 'e'
    <B> -> 'b'
            """,
            {"A", "B"},
        ),
        (
            """
    <E> -> <E> '+' <E>
    <E> -> <E> '*' <E>
    <E> -> 'a'
            """,
            {"E"},
        ),
        (
            """
    <E> -> <E> '+' <T> 
    <E> -> <T>
    <T> -> <T> '*' <F>
    <T> -> <F>
    <F> -> 'id'
            """,
            {"E", "T"},
        ),
        (
            """
    <S> -> <S> '0' <S> '1' <S>
    <S> -> '0' '1'
            """,
            {"S"},
        ),
        (
            """
    <S> -> '(' <L> ')'
    <S> -> 'a'
    <L> -> <L> ',' <S>
    <L> -> <S>
            """,
            {"L"},
        ),
        (
            """
    <X> -> <X> <S> 'b'
    <X> -> <S> 'a'
    <X> -> 'b'
    <S> -> <S> 'b'
    <S> -> <X> 'a'
    <S> -> 'a'
        """,
            {"X", "S"},
        ),
    ],
)
def test_left_recursion(grammar_str, expected):
    grammar = Grammar.from_str(grammar_str)
    actual = {nt.name for nt in compute_left_recursion_non_terminals(grammar)}
    assert actual == expected
