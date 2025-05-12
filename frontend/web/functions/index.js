const functions = require('firebase-functions');
const axios = require('axios');

exports.generateText = functions.https.onCall(async (data, context) => {
  if (!context.auth) {
    throw new functions.https.HttpsError('unauthenticated', 'User must be authenticated');
  }

  try {
    const response = await axios.post(
      'http://localhost:8000/process_input', // Update to production URL
      {
        text: data.text,
        user_id: context.auth.uid,
        session_id: data.session_id
      },
      {
        headers: { Authorization: 'Bearer dummy_token' } // Replace with Firebase ID token
      }
    );
    return response.data;
  } catch (error) {
    throw new functions.https.HttpsError('internal', error.message);
  }
});