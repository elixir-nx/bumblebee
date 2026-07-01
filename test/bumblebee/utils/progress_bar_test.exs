defmodule Bumblebee.Utils.ProgressBarTest do
  use ExUnit.Case, async: true

  alias Bumblebee.Utils.ProgressBar
  alias ExUnit.CaptureIO

  test "renders various widths of progress bars" do
    assert "\r|================================                                 | 50% 0.2/0.4" ==
             CaptureIO.capture_io(fn -> ProgressBar.render(0.2, 0.4) end)

    assert "\r|                                                                    |  0% 0/10" ==
             CaptureIO.capture_io(fn -> ProgressBar.render(0, 10) end)

    assert "\r|==================================                                  | 50% 5/10" ==
             CaptureIO.capture_io(fn -> ProgressBar.render(5, 10) end)

    assert "\r|==============================================================    | 95% 9.5/10" ==
             CaptureIO.capture_io(fn -> ProgressBar.render(9.5, 10) end)

    assert "\r|=============================================================== | 99% 9.999/10" ==
             CaptureIO.capture_io(fn -> ProgressBar.render(9.999, 10) end)

    assert "\r|===================================================================|100% 10/10" ==
             CaptureIO.capture_io(fn -> ProgressBar.render(10, 10) end)
  end

  test "renders bars when unexpected inputs given" do
    assert "\r|                                                                  |  0% -10/10" ==
             CaptureIO.capture_io(fn -> ProgressBar.render(-10, 10) end)

    assert "\r|===================================================================|100% 20/10" ==
             CaptureIO.capture_io(fn -> ProgressBar.render(20, 10) end)
  end

  test "formats byte counts" do
    assert "\r|======                                                           | 10% 10/100B" ==
             CaptureIO.capture_io(fn -> ProgressBar.render(10, 100, unit: :bytes) end)

    assert "\r|=================================                                  | 50% 1/2KB" ==
             CaptureIO.capture_io(fn -> ProgressBar.render(1_000, 2_000, unit: :bytes) end)

    assert "\r|=================================                                  | 50% 1/2MB" ==
             CaptureIO.capture_io(fn ->
               ProgressBar.render(1_000_000, 2_000_000, unit: :bytes)
             end)
  end
end
