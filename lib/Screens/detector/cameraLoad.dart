// import 'package:flutter/material.dart';
// import 'package:camera/camera.dart';
// import 'package:permission_handler/permission_handler.dart';
// import 'package:tflite_flutter/tflite_flutter.dart';
//
// class CameraLoad extends StatefulWidget {
//   @override
//   _CameraLoadState createState() => _CameraLoadState();
// }
//
// class _CameraLoadState extends State<CameraLoad> {
//   late CameraController _controller;
//   late Future<void> _initializeControllerFuture;
//   late Interpreter _interpreter;
//
//   @override
//   void initState() {
//     super.initState();
//     _checkPermissionAndInitializeCamera();
//     _loadModel();
//   }
//
//   // Check camera permissions and initialize the camera
//   void _checkPermissionAndInitializeCamera() async {
//     // Check if camera permission is granted
//     if (await Permission.camera.isGranted) {
//       // Permission is already granted, initialize the camera
//       _initializeCamera();
//     } else {
//       // Request camera permission
//       var status = await Permission.camera.request();
//       if (status.isGranted) {
//         // Permission granted, initialize the camera
//         _initializeCamera();
//       } else {
//         // Permission denied, handle accordingly (e.g., show a message)
//         print('Camera permission denied');
//       }
//     }
//   }
//
//   // Initialize the camera when permission is granted
//   void _initializeCamera() async {
//     List<CameraDescription> cameras = await availableCameras();
//     CameraDescription? frontCamera = cameras.firstWhere(
//           (camera) => camera.lensDirection == CameraLensDirection.front,
//       orElse: () => cameras.first,
//     );
//
//     // Only initialize the controller if permission is granted
//     _controller = CameraController(frontCamera, ResolutionPreset.medium);
//     _initializeControllerFuture = _controller.initialize();
//     setState(() {}); // Rebuild the UI to display the camera preview
//   }
//
//   Future<void> _loadModel() async {
//     _interpreter = await Interpreter.fromAsset('my_model.tflite');
//   }
//
//   @override
//   void dispose() {
//     _controller.dispose();
//     _interpreter.dispose();
//     super.dispose();
//   }
//
//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       floatingActionButton: FloatingActionButton(
//         onPressed: () {
//           Navigator.pushNamed(context, 'endWorkout');
//         },
//         child: Icon(Icons.exit_to_app),
//       ),
//       body: Column(
//         children: [
//           FutureBuilder<void>(
//             future: _initializeControllerFuture,
//             builder: (context, snapshot) {
//               if (snapshot.connectionState == ConnectionState.done) {
//                 return CameraPreview(_controller);
//               } else {
//                 return Center(child: CircularProgressIndicator());
//               }
//             },
//           ),
//         ],
//       ),);
//   }
// }