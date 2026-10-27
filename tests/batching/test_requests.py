import pytest

from mlserver.types import (
    InferenceRequest,
    RequestInput,
    ResponseOutput,
    InferenceResponse,
    Parameters,
)
from mlserver.batching.requests import BatchedRequests, _get_parameters


@pytest.mark.parametrize(
    "value, expected",
    [
        ("123", {"foo": ["123"]}),
        (["first", "second"], {"foo": ["first", "second"]}),
        (("first", "second"), {"foo": ["first", "second"]}),
    ],
)
def test_get_parameters_preserves_scalar_values(value, expected):
    response_output = ResponseOutput(
        name="foo",
        datatype="INT32",
        shape=[1],
        data=[1],
        parameters=Parameters(foo=value),
    )

    assert dict(_get_parameters(response_output)) == expected


@pytest.mark.parametrize("value", ["123", 123, False])
@pytest.mark.parametrize("batch_sizes", [(1, 1), (2, 1), (1, 2)])
def test_split_response_preserves_scalar_parameter_for_each_request(value, batch_sizes):
    first_data = list(range(batch_sizes[0]))
    second_data = list(range(batch_sizes[0], sum(batch_sizes)))
    requests = {
        "request-1": InferenceRequest(
            id="request-1",
            inputs=[
                RequestInput(
                    name="input",
                    datatype="INT32",
                    shape=[len(first_data)],
                    data=first_data,
                )
            ],
        ),
        "request-2": InferenceRequest(
            id="request-2",
            inputs=[
                RequestInput(
                    name="input",
                    datatype="INT32",
                    shape=[len(second_data)],
                    data=second_data,
                )
            ],
        ),
    }
    response = InferenceResponse(
        model_name="test-model",
        outputs=[
            ResponseOutput(
                name="output",
                datatype="INT32",
                shape=[sum(batch_sizes)],
                data=first_data + second_data,
                parameters=Parameters(foo=value),
            )
        ],
    )

    split = BatchedRequests(requests).split_response(response)

    assert split["request-1"].outputs[0].parameters.foo == value
    assert split["request-2"].outputs[0].parameters.foo == value
    assert type(split["request-1"].outputs[0].parameters.foo) is type(value)
    assert type(split["request-2"].outputs[0].parameters.foo) is type(value)
    assert split["request-1"].outputs[0].data.root == first_data
    assert split["request-2"].outputs[0].data.root == second_data
    assert split["request-1"].outputs[0].shape == [len(first_data)]
    assert split["request-2"].outputs[0].shape == [len(second_data)]


@pytest.mark.parametrize("batch_sizes", [(2, 1, 1), (1, 2, 1), (2, 3, 2)])
def test_split_response_matches_parameters_to_requests(
    batch_sizes: tuple[int, int, int],
) -> None:
    requests = {
        f"request-{idx}": InferenceRequest(
            id=f"external-{idx}",
            inputs=[
                RequestInput(
                    name="input",
                    datatype="INT32",
                    shape=[batch_size, 2],
                    data=list(range(idx * 10, idx * 10 + batch_size * 2)),
                    parameters=Parameters(
                        tag=f"tag-{idx}",
                        content_type="np",
                        headers={"x-test": "batch"},
                    ),
                )
            ],
        )
        for idx, batch_size in enumerate(batch_sizes)
    }
    batched = BatchedRequests(requests)
    response = InferenceResponse(
        model_name="test-model",
        outputs=[ResponseOutput(**batched.merged_request.inputs[0].model_dump())],
    )

    split = batched.split_response(response)

    for internal_id, request in requests.items():
        output = split[internal_id].outputs[0]
        request_input = request.inputs[0]
        assert split[internal_id].id == request.id
        assert output.parameters == request_input.parameters
        assert output.data == request_input.data
        assert output.shape == request_input.shape


@pytest.mark.parametrize(
    "value",
    [
        "123",
        123,
        False,
        ["123", "456"],
        [123, 456],
        [],
        {},
        {"key": 123},
        {"nested": {"values": []}},
    ],
)
def test_split_response_preserves_parameter_for_single_request(value):
    requests = {
        "request-1": InferenceRequest(
            id="request-1",
            inputs=[
                RequestInput(name="input", datatype="INT32", shape=[2], data=[1, 2])
            ],
        )
    }
    response = InferenceResponse(
        model_name="test-model",
        outputs=[
            ResponseOutput(
                name="output",
                datatype="INT32",
                shape=[2],
                data=[1, 2],
                parameters=Parameters(foo=value),
            )
        ],
    )

    split = BatchedRequests(requests).split_response(response)

    assert split["request-1"].outputs[0].parameters.foo == value


@pytest.mark.parametrize(
    "request_inputs, expected_request_input, expected_minibatch_sizes",
    [
        (
            {
                "req-1": RequestInput(
                    name="foo",
                    datatype="INT32",
                    shape=[1, 3],
                    data=[1, 2, 3],
                    parameters=Parameters(content_type="np"),
                ),
                "req-2": RequestInput(
                    name="foo",
                    datatype="INT32",
                    shape=[1, 3],
                    data=[4, 5, 6],
                ),
                "req-3": RequestInput(
                    name="foo",
                    datatype="INT32",
                    shape=[1, 3],
                    data=[7, 8, 9],
                    parameters=Parameters(foo="bar"),
                ),
            },
            RequestInput(
                name="foo",
                datatype="INT32",
                shape=[3, 3],
                data=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                parameters=Parameters(content_type="np", foo="bar"),
            ),
            {"req-1": 1, "req-2": 1, "req-3": 1},
        ),
        (
            {
                "req-1": RequestInput(
                    name="foo",
                    datatype="INT32",
                    shape=[1, 3],
                    data=[1, 2, 3],
                    parameters=Parameters(content_type="np", foo="bar"),
                ),
                "req-2": RequestInput(
                    name="foo",
                    datatype="INT32",
                    shape=[1, 3],
                    data=[4, 5, 6],
                ),
                "req-3": RequestInput(
                    name="foo",
                    datatype="INT32",
                    shape=[1, 3],
                    data=[7, 8, 9],
                    parameters=Parameters(foo="bar", time="13:00"),
                ),
            },
            RequestInput(
                name="foo",
                datatype="INT32",
                shape=[3, 3],
                data=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                parameters=Parameters(
                    content_type="np", foo=["bar", "bar"], time="13:00"
                ),
            ),
            {"req-1": 1, "req-2": 1, "req-3": 1},
        ),
        (
            {
                "req-1": RequestInput(
                    name="foo",
                    datatype="INT32",
                    shape=[2, 3],
                    data=[1, 2, 3, 10, 11, 12],
                ),
                "req-2": RequestInput(
                    name="foo", datatype="INT32", shape=[1, 3], data=[4, 5, 6]
                ),
                "req-3": RequestInput(
                    name="foo", datatype="INT32", shape=[1, 3], data=[7, 8, 9]
                ),
            },
            RequestInput(
                name="foo",
                datatype="INT32",
                shape=[4, 3],
                data=[1, 2, 3, 10, 11, 12, 4, 5, 6, 7, 8, 9],
            ),
            {"req-1": 2, "req-2": 1, "req-3": 1},
        ),
        (
            {
                "req-1": RequestInput(
                    name="foo", datatype="INT32", shape=[1, 3], data=[[1, 2, 3]]
                ),
                "req-2": RequestInput(
                    name="foo", datatype="INT32", shape=[1, 3], data=[[4, 5, 6]]
                ),
                "req-3": RequestInput(
                    name="foo", datatype="INT32", shape=[1, 3], data=[[7, 8, 9]]
                ),
            },
            RequestInput(
                name="foo",
                datatype="INT32",
                shape=[3, 3],
                data=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            ),
            {"req-1": 1, "req-2": 1, "req-3": 1},
        ),
        (
            {
                "req-1": RequestInput(
                    name="foo", datatype="BYTES", shape=[1, 3], data=b"abc"
                ),
                "req-2": RequestInput(
                    name="foo", datatype="BYTES", shape=[1, 3], data=b"def"
                ),
                "req-3": RequestInput(
                    name="foo", datatype="BYTES", shape=[1, 3], data=b"ghi"
                ),
            },
            RequestInput(
                name="foo",
                datatype="BYTES",
                shape=[3, 3],
                data=b"abcdefghi",
            ),
            {"req-1": 1, "req-2": 1, "req-3": 1},
        ),
        (
            {
                "req-1": RequestInput(
                    name="foo", datatype="BYTES", shape=[1, 3], data="abc"
                ),
                "req-2": RequestInput(
                    name="foo", datatype="BYTES", shape=[1, 3], data="def"
                ),
                "req-3": RequestInput(
                    name="foo", datatype="BYTES", shape=[1, 3], data="ghi"
                ),
            },
            RequestInput(
                name="foo",
                datatype="BYTES",
                shape=[3, 3],
                data="abcdefghi",
            ),
            {"req-1": 1, "req-2": 1, "req-3": 1},
        ),
    ],
)
def test_merge_request_inputs(
    request_inputs: dict[str, RequestInput],
    expected_request_input: RequestInput,
    expected_minibatch_sizes: dict[str, int],
):
    batched = BatchedRequests()
    merged = batched._merge_request_inputs(request_inputs)

    assert merged == expected_request_input
    assert batched._minibatch_sizes == expected_minibatch_sizes


@pytest.mark.parametrize(
    "inference_requests, expected",
    [
        (
            {
                "req-1": InferenceRequest(
                    parameters=Parameters(content_type="np"),
                    inputs=[
                        RequestInput(
                            name="foo", datatype="INT32", data=[1, 2, 3], shape=[1, 3]
                        )
                    ],
                ),
                "req-2": InferenceRequest(
                    parameters=Parameters(foo="bar"),
                    inputs=[
                        RequestInput(
                            name="foo", datatype="INT32", data=[4, 5, 6], shape=[1, 3]
                        )
                    ],
                ),
            },
            InferenceRequest(
                parameters=Parameters(content_type="np", foo="bar"),
                inputs=[
                    RequestInput(
                        name="foo",
                        datatype="INT32",
                        data=[1, 2, 3, 4, 5, 6],
                        shape=[2, 3],
                    )
                ],
            ),
        ),
        (
            {
                "req-1": InferenceRequest(
                    inputs=[
                        RequestInput(
                            name="foo", datatype="INT32", data=[1, 2, 3], shape=[1, 3]
                        ),
                        RequestInput(
                            name="bar", datatype="BYTES", data=b"abc", shape=[1, 3]
                        ),
                    ]
                ),
                "req-2": InferenceRequest(
                    inputs=[
                        RequestInput(
                            name="foo", datatype="INT32", data=[4, 5, 6], shape=[1, 3]
                        ),
                        RequestInput(
                            name="bar", datatype="BYTES", data=b"def", shape=[1, 3]
                        ),
                    ]
                ),
            },
            InferenceRequest(
                inputs=[
                    RequestInput(
                        name="foo",
                        datatype="INT32",
                        data=[1, 2, 3, 4, 5, 6],
                        shape=[2, 3],
                    ),
                    RequestInput(
                        name="bar",
                        datatype="BYTES",
                        data=b"abcdef",
                        shape=[2, 3],
                    ),
                ]
            ),
        ),
    ],
)
def test_merged_request(
    inference_requests: dict[str, InferenceRequest],
    expected: InferenceRequest,
):
    batched = BatchedRequests(inference_requests)
    merged_request = batched.merged_request

    assert merged_request == expected


@pytest.mark.parametrize(
    "minibatch_sizes, response_output, expected",
    [
        (
            {"req-1": 1, "req-2": 1, "req-3": 1},
            ResponseOutput(
                name="foo",
                datatype="INT32",
                shape=[3, 3],
                data=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                parameters=Parameters(
                    content_type="np",
                    foo=["foo_1", "foo_2"],
                    bar=["bar_1", "bar_2", "bar_3"],
                ),
            ),
            [
                ResponseOutput(
                    name="foo",
                    datatype="INT32",
                    shape=[1, 3],
                    data=[1, 2, 3],
                    parameters=Parameters(content_type="np", foo="foo_1", bar="bar_1"),
                ),
                ResponseOutput(
                    name="foo",
                    datatype="INT32",
                    shape=[1, 3],
                    data=[4, 5, 6],
                    parameters=Parameters(content_type="np", foo="foo_2", bar="bar_2"),
                ),
                ResponseOutput(
                    name="foo",
                    datatype="INT32",
                    shape=[1, 3],
                    data=[7, 8, 9],
                    parameters=Parameters(content_type="np", bar="bar_3"),
                ),
            ],
        ),
        (
            {"req-1": 1, "req-2": 1, "req-3": 1},
            ResponseOutput(
                name="foo",
                datatype="INT32",
                shape=[3, 3],
                data=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                parameters=Parameters(content_type="np"),
            ),
            [
                ResponseOutput(
                    name="foo",
                    datatype="INT32",
                    shape=[1, 3],
                    data=[1, 2, 3],
                    parameters=Parameters(content_type="np"),
                ),
                ResponseOutput(
                    name="foo",
                    datatype="INT32",
                    shape=[1, 3],
                    data=[4, 5, 6],
                    parameters=Parameters(content_type="np"),
                ),
                ResponseOutput(
                    name="foo",
                    datatype="INT32",
                    shape=[1, 3],
                    data=[7, 8, 9],
                    parameters=Parameters(content_type="np"),
                ),
            ],
        ),
        (
            {"req-1": 1, "req-2": 1, "req-3": 1},
            ResponseOutput(
                name="foo",
                datatype="BYTES",
                shape=[3, 3],
                data=b"abcdefghi",
            ),
            [
                ResponseOutput(name="foo", datatype="BYTES", shape=[1, 3], data=b"abc"),
                ResponseOutput(name="foo", datatype="BYTES", shape=[1, 3], data=b"def"),
                ResponseOutput(name="foo", datatype="BYTES", shape=[1, 3], data=b"ghi"),
            ],
        ),
        (
            {"req-1": 1, "req-2": 2, "req-3": 1},
            ResponseOutput(
                name="foo",
                datatype="BYTES",
                shape=[4, 3],
                data=b"abcdefjklghi",
            ),
            [
                ResponseOutput(name="foo", datatype="BYTES", shape=[1, 3], data=b"abc"),
                ResponseOutput(
                    name="foo", datatype="BYTES", shape=[2, 3], data=b"defjkl"
                ),
                ResponseOutput(name="foo", datatype="BYTES", shape=[1, 3], data=b"ghi"),
            ],
        ),
    ],
)
def test_split_response_output(
    minibatch_sizes: dict[str, int],
    response_output: ResponseOutput,
    expected: list[ResponseOutput],
):
    batched = BatchedRequests()
    batched._minibatch_sizes = minibatch_sizes
    split = batched._split_response_output(response_output)

    assert list(split.values()) == expected


@pytest.mark.parametrize(
    "inference_requests, inference_response, expected",
    [
        (
            {
                "query-1": InferenceRequest(
                    id="query-1",
                    inputs=[
                        RequestInput(
                            name="bli", datatype="FP32", shape=[1, 1], data=[0]
                        )
                    ],
                ),
                "query-2": InferenceRequest(
                    id="query-2",
                    inputs=[
                        RequestInput(
                            name="bli", datatype="FP32", shape=[1, 1], data=[0]
                        )
                    ],
                ),
            },
            InferenceResponse(
                model_name="sum-model",
                parameters=Parameters(foo="bar"),
                outputs=[
                    ResponseOutput(
                        name="foo",
                        datatype="INT32",
                        shape=[2, 2],
                        data=[1, 2, 3, 4],
                    ),
                    ResponseOutput(
                        name="bar",
                        datatype="BYTES",
                        shape=[2, 3],
                        data=b"abcdef",
                    ),
                ],
            ),
            [
                InferenceResponse(
                    id="query-1",
                    model_name="sum-model",
                    parameters=Parameters(foo="bar"),
                    outputs=[
                        ResponseOutput(
                            name="foo",
                            datatype="INT32",
                            shape=[1, 2],
                            data=[1, 2],
                        ),
                        ResponseOutput(
                            name="bar",
                            datatype="BYTES",
                            shape=[1, 3],
                            data=b"abc",
                        ),
                    ],
                ),
                InferenceResponse(
                    id="query-2",
                    model_name="sum-model",
                    parameters=Parameters(foo="bar"),
                    outputs=[
                        ResponseOutput(
                            name="foo",
                            datatype="INT32",
                            shape=[1, 2],
                            data=[3, 4],
                        ),
                        ResponseOutput(
                            name="bar",
                            datatype="BYTES",
                            shape=[1, 3],
                            data=b"def",
                        ),
                    ],
                ),
            ],
        )
    ],
)
def test_split_response(
    inference_requests: dict[str, InferenceRequest],
    inference_response: InferenceResponse,
    expected: list[InferenceResponse],
):
    batched = BatchedRequests(inference_requests)

    responses = batched.split_response(inference_response)

    assert inference_requests.keys() == responses.keys()
    assert list(responses.values()) == expected
