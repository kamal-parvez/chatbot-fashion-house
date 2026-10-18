"""Query routing: FAQ-vs-Product classification and task-nature detection."""

from typing import Dict, Tuple

from opentelemetry.trace import Status, StatusCode

from app import config, llm
from app.tracing import get_tracer

tracer = get_tracer()


def check_if_faq_or_product(query: str) -> Tuple[str, int]:
    """Labels a query as 'FAQ', 'Product', or 'undefined'."""
    prompt = f"""Label the following instruction as an FAQ related answer or a product related answer for a clothing store.
Product related answers are answers specific about product information or that needs to use the products to give an answer.
Examples:
        Is there a refund for incorrectly bought clothes? Label: FAQ
        Where are your stores located?: Label: FAQ
        Tell me about the cheapest T-shirts that you have. Label: Product
        Do you have blue T-shirts under 100 dollars? Label: Product
        What are the available sizes for the t-shirts? Label: FAQ
        How can I contact you via phone? Label: FAQ
        How can I find the promotions? Label: FAQ
        Give me ideas for a sunny look. Label: Product
Return only one of the two labels: FAQ or Product, nothing more.
Query to classify: {query}
         """

    with tracer.start_as_current_span("routing_faq_or_product", openinference_span_kind="tool") as span:
        span.set_input({"query": query})

        text, total_tokens = llm.generate(
            prompt, temperature=0, max_tokens=10, model=config.GEMINI_ROUTER_MODEL
        )

        label_lower = text.lower()
        if "faq" in label_lower:
            label = "FAQ"
        elif "product" in label_lower:
            label = "Product"
        else:
            label = "undefined"

        span.set_output({"label": label, "total_tokens": total_tokens})
        span.set_status(Status(StatusCode.OK))

    return label, total_tokens


def decide_task_nature(query: str) -> Tuple[str, int]:
    """Labels a query as 'creative' or 'technical'."""
    prompt = f"""Decide if the following query is a query that requires creativity (creating, composing, making new things) or technical (information about products, prices etc.). Label it as creative or technical.
  Examples:
  Give me suggestions on a nice look for a nightclub. Label: creative
  What are the blue dresses you have available? Label: technical
  Give me three Tshirts for summer. Label: technical
  Give me a look for attending a wedding party. Label: creative
  Give me suggestions on clothes that match a green Tshirt. Label: creative
  I would like a suggestion on which products match a green Tshirt I already have. Label: creative

  Query to be analyzed: {query}. Only output one token with the label
  """

    with tracer.start_as_current_span("decide_task_nature", openinference_span_kind="tool") as span:
        span.set_input({"query": query})

        text, total_tokens = llm.generate(
            prompt, temperature=0, max_tokens=5, model=config.GEMINI_ROUTER_MODEL
        )
        label = text.strip().lower()

        span.set_output({"label": label, "total_tokens": total_tokens})
        span.set_status(Status(StatusCode.OK))

    return label, total_tokens


def get_params_for_task(task: str) -> Dict[str, float]:
    """Sampling parameters tailored to the task nature."""
    parameters_dict = {
        "creative": {"top_p": 0.9, "temperature": 1.0},
        "technical": {"top_p": 0.7, "temperature": 0.3},
    }
    return parameters_dict.get(task, {"top_p": 0.5, "temperature": 1.0})
