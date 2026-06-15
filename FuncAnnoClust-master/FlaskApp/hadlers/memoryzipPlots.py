import io
import zipfile
import plotly.io as pio
import plotly.graph_objects as go  # Import go
from kaleido.scopes.plotly import PlotlyScope
import base64

def export_plots(data, export_format):

    plots_dict = data.getPlots()
    plots_to_zip = {}

    # Validate export format
    supported_formats = ["png", "jpeg", "svg", "pdf"]
    if export_format not in supported_formats:
        raise ValueError(f"Unsupported export format: {export_format}. Supported formats: {supported_formats}")

    for name_plots in plots_dict:
        try:
            # Deserialize JSON string into Plotly figure
            exported_plot = pio.from_json(plots_dict[name_plots])
        except Exception as e:
            print(f"Error deserializing plot '{name_plots}': {e}")
            continue

        # Convert plot to image
        buffer_img = io.BytesIO()
        print("boo_4")
        try:
            if export_format == "svg":
                fig_bytes = exported_plot.to_image(format=export_format)  # Get SVG as bytes
                buffer_img.write(fig_bytes)  # Write bytes to buffer
            else:
                scope = PlotlyScope()
                img_bytes = scope.transform(figure=exported_plot, format=export_format)
                buffer_img.write(img_bytes)
            buffer_img.seek(0)  # Reset buffer position

        except Exception as e:
            print(f"Error converting plot '{name_plots}' to image: {e}")
            continue

        # Add to dictionary
        plot_file = f"{name_plots}.{export_format}"
        plots_to_zip[plot_file] = buffer_img

    print("booo11", plots_to_zip)
    return plots_to_zip


def get_zip_buffer(data, export_format):

    zip_buffer = io.BytesIO()

    # Export plots
    print("boo")
    plots_to_zip = export_plots(data, export_format)
    print("_2")
    print(plots_to_zip)

    # Create ZIP archive
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for file_name, buffer in plots_to_zip.items():
            print("boo_3")
            buffer.seek(0)  # Reset buffer position
            zip_file.writestr(file_name, buffer.read())

    zip_buffer.seek(0)  # Reset ZIP buffer position
    return zip_buffer
