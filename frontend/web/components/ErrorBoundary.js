import { Component } from 'react';
import { Box, Text } from '@chakra-ui/react';

class ErrorBoundary extends Component {
  state = { hasError: false };

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    console.error('ErrorBoundary caught:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <Box p={8} textAlign="center" color="white" bg="gray.900">
          <Text fontSize="xl">Something went wrong. Please try again later.</Text>
        </Box>
      );
    }
    return this.props.children;
  }
}

export default ErrorBoundary;
