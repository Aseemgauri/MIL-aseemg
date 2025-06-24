// // Creating GStreamer objects to manage the pipeline.
// GstElement *pipeline, *appsrc;

// // Helper function to initialize Gstreamer pipeline to send the audio data.
// void init_gstreamer_pipeline() {
//   std::cout << "Inside init for pipeline";
//   gst_init(nullptr, nullptr);

//   // pipeline = gst_parse_launch(
//   //     "appsrc name=mysource format=time is-live=true do-timestamp=true "
//   //     "! audioconvert ! audioresample ! opusenc ! udpsink host=127.0.0.1 port=4200",
//   //     nullptr
//   // );


//   // Experimenting with uncompressed audio.
//   pipeline = gst_parse_launch(
//       "appsrc name=mysource format=time is-live=true do-timestamp=true "
//       "! audioconvert ! audioresample ! udpsink host=127.0.0.1 port=4200",
//       nullptr
//   );


//   appsrc = gst_bin_get_by_name(GST_BIN(pipeline), "mysource");

//   // Specifying the type of audio data we are sending.
//   GstCaps *caps = gst_caps_new_simple(
//     "audio/x-raw",
//     "format", G_TYPE_STRING, "F32LE",  // You are passing float data
//     "channels", G_TYPE_INT, 1,
//     "rate", G_TYPE_INT, 44100,
//     NULL
//   );
//   gst_app_src_set_caps(GST_APP_SRC(appsrc), caps);
//   gst_caps_unref(caps);

//   gst_element_set_state(pipeline, GST_STATE_PLAYING);
//   std::cout << "GStreamer pipeline initialized.\n";
// }

// /*----- Core Functions ----- */

// // Function responsible for sending the actual one second buffer of data
// // to the python server.
// void send_float_buf(const float* buf, int len) {
//   // Creating a buffer in Gstreamer to store our audio data
//   // and preparing it to be written to.
//   GstBuffer* gstBuf = gst_buffer_new_allocate(nullptr, len * sizeof(float), nullptr);
//   GstMapInfo map;
//   gst_buffer_map(gstBuf, &map, GST_MAP_WRITE);

//   // Copying data from our buffer into the Gstreamer buffer.
//   // Capping off the buffer to prevent further writing.
//   memcpy(map.data, buf, len * sizeof(float));
//   gst_buffer_unmap(gstBuf, &map);

//   GstStateChangeReturn res = gst_element_set_state(pipeline, GST_STATE_PLAYING);
//   if (res == GST_STATE_CHANGE_FAILURE) {
//       std::cerr << "Failed to start GStreamer pipeline.\n";
//       exit(1);
//   }

//   // Attempting to send sound to machine created in init step.
//   // Error-checking as need be.
//   GstFlowReturn ret;
//   g_signal_emit_by_name(appsrc, "push-buffer", gstBuf, &ret);
//   if (ret != GST_FLOW_OK) {
//     std::cerr << "Failed to push buffer to GStreamer pipeline.\n";
//   } else {
//     std::cout << "Pushed 1-second buffer to GStreamer.\n";
//   }

//   // Cleaning up the buffer after successfully pushing the data
//   // to the pipeline.
//   gst_buffer_unref(gstBuf);
// }