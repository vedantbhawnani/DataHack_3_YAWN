// File generated by FlutterFire CLI.
// ignore_for_file: lines_longer_than_80_chars, avoid_classes_with_only_static_members
import 'package:firebase_core/firebase_core.dart' show FirebaseOptions;
import 'package:flutter/foundation.dart'
    show defaultTargetPlatform, kIsWeb, TargetPlatform;

class DefaultFirebaseOptions {
  static FirebaseOptions get currentPlatform {
    if (kIsWeb) {
      return web;
    }
    switch (defaultTargetPlatform) {
      case TargetPlatform.android:
        return android;
      case TargetPlatform.iOS:
        return ios;
      case TargetPlatform.macOS:
        return macos;
      case TargetPlatform.windows:
        throw UnsupportedError(
          'DefaultFirebaseOptions have not been configured for windows - '
          'you can reconfigure this by running the FlutterFire CLI again.',
        );
      case TargetPlatform.linux:
        throw UnsupportedError(
          'DefaultFirebaseOptions have not been configured for linux - '
          'you can reconfigure this by running the FlutterFire CLI again.',
        );
      default:
        throw UnsupportedError(
          'DefaultFirebaseOptions are not supported for this platform.',
        );
    }
  }

  static const FirebaseOptions web = FirebaseOptions(
    apiKey: 'AIzaSyBujFIY172JOSBQoSLGGTmn2n0LS0nAqxI',
    appId: '1:317547619587:web:daa2bc597ca3cf4c503536',
    messagingSenderId: '317547619587',
    projectId: 'fitai-6ada4',
    authDomain: 'fitai-6ada4.firebaseapp.com',
    storageBucket: 'fitai-6ada4.appspot.com',
    measurementId: 'G-CCTTRX11S7',
  );

  static const FirebaseOptions android = FirebaseOptions(
    apiKey: 'AIzaSyAO0fuu6U13LQFDmy9yJHswMS1N0uD5FWU',
    appId: '1:317547619587:android:cfa35f46f7ffc7c9503536',
    messagingSenderId: '317547619587',
    projectId: 'fitai-6ada4',
    storageBucket: 'fitai-6ada4.appspot.com',
  );

  static const FirebaseOptions ios = FirebaseOptions(
    apiKey: 'AIzaSyDzCHmY49kSAegd-phn4bvWQo9YCM-yW8g',
    appId: '1:317547619587:ios:bdf73033237cb5d2503536',
    messagingSenderId: '317547619587',
    projectId: 'fitai-6ada4',
    storageBucket: 'fitai-6ada4.appspot.com',
    iosBundleId: 'com.example.exerciserecognition',
  );

  static const FirebaseOptions macos = FirebaseOptions(
    apiKey: 'AIzaSyDzCHmY49kSAegd-phn4bvWQo9YCM-yW8g',
    appId: '1:317547619587:ios:84ba288f2c6d61e9503536',
    messagingSenderId: '317547619587',
    projectId: 'fitai-6ada4',
    storageBucket: 'fitai-6ada4.appspot.com',
    iosBundleId: 'com.example.exerciserecognition.RunnerTests',
  );
}