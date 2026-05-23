package com.example.tensorflow_demo

// WakeWordService has been removed.
//
// A background Android SpeechRecognizer foreground service inherently holds
// the microphone open in ALL apps (Instagram, WhatsApp, etc.), which is
// intrusive for users. It cannot be made to behave like Siri/Bixby/Google
// Assistant, which use dedicated low-power hardware chips baked into the OS.
//
// Wake-word detection ("Envision") is instead handled in-app via Flutter STT
// polling, which only runs while the Envision app is in the foreground and
// automatically releases the mic when the user switches to another app.
