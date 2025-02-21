from timbal import Flow


def identity_handler(x):
    return x


def test_compile():
    flow = (
        Flow(id="test_compile")
        .add_step("1", identity_handler)
        .add_step("2", identity_handler)
        .add_link("1", "2")
    )

    flow.compile()

    assert hasattr(flow, '_dag')
    assert hasattr(flow, '_rev_dag')

    assert hasattr(flow, '_params_model')
    assert hasattr(flow, '_return_model')
    assert hasattr(flow, '_params_model_schema')
    assert hasattr(flow, '_return_model_schema')
    