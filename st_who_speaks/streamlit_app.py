import warnings

warnings.filterwarnings(
    "ignore",
    message=r"Module 'speechbrain\.pretrained' was deprecated, redirecting to 'speechbrain\.inference'\. Please update your script\. This is a change from SpeechBrain 1\.0\.",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"`torch\.cuda\.amp\.custom_fwd\(args\.\.\.\)` is deprecated\. Please use `torch\.amp\.custom_fwd\(args\.\.\., device_type='cuda'\)` instead\.",
    category=FutureWarning,
)


def main() -> None:
    from st_who_speaks.app import main as run_app

    run_app()


if __name__ == "__main__":
    main()
