"""Calibration prompt sets across task categories (Blueprint §30)."""
from __future__ import annotations

CATEGORIES = {
    "coding": [
        "Debug this FastAPI JWT authentication bug: the token expires before refresh.",
        "Implement Dijkstra in Rust using a binary heap.",
        "Write a Python function to reverse a linked list in place.",
        "Explain the difference between a SQL inner join and left join.",
        "Why does my C++ program segfault when freeing a pointer twice?",
    ],
    "math": [
        "Solve: x^2 - 5x + 6 = 0.",
        "Prove that the square root of 2 is irrational.",
        "Compute the derivative of sin(x)*e^x.",
        "What is the sum of the first 100 positive integers?",
        "Explain dynamic programming using the coin change problem.",
    ],
    "reasoning": [
        "If all bloops are razzies and all razzies are lazzies, are all bloops lazzies?",
        "A farmer has 17 sheep and all but 9 run away. How many remain?",
        "Explain why a red herring fallacy weakens an argument.",
        "Order these: losing keys, finding keys, locking door, leaving house.",
        "Which is heavier: a kilogram of steel or two pounds of feathers?",
    ],
    "general_knowledge": [
        "What is the capital of Australia?",
        "Describe the causes of World War I.",
        "Explain photosynthesis in simple terms.",
        "Who wrote Pride and Prejudice?",
        "What is the boiling point of water at sea level?",
    ],
    "writing": [
        "Write a haiku about autumn rain.",
        "Draft a polite email declining a meeting invitation.",
        "Compose a short poem about the ocean.",
        "Write a product description for a wooden coffee mug.",
        "Summarize the plot of Hamlet in three sentences.",
    ],
}


def all_prompts():
    return [(cat, p) for cat, ps in CATEGORIES.items() for p in ps]
