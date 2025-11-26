import React, { useState, useEffect } from 'react';
import {
    Box,
    VStack,
    HStack,
    Text,
    Button,
    Textarea,
    Badge,
    Divider,
    useToast,
    Heading,
} from '@chakra-ui/react';
import axios from 'axios';

const ReviewInterface = ({ datasetId }) => {
    const [examples, setExamples] = useState([]);
    const [currentIndex, setCurrentIndex] = useState(0);
    const [comment, setComment] = useState('');
    const toast = useToast();

    useEffect(() => {
        if (datasetId) {
            fetchExamples();
        }
    }, [datasetId]);

    const fetchExamples = async () => {
        try {
            const response = await axios.get(`http://localhost:8000/api/datasets/${datasetId}/examples?limit=10`);
            setExamples(response.data.examples || []);
        } catch (error) {
            console.error(error);
        }
    };

    const handleApprove = async () => {
        // Mock approval
        toast({ title: 'Example approved', status: 'success' });
        nextExample();
    };

    const handleReject = async () => {
        // Mock rejection
        toast({ title: 'Example rejected', status: 'warning' });
        nextExample();
    };

    const handleComment = async () => {
        if (!comment) return;
        // Mock comment submission
        toast({ title: 'Comment added', status: 'info' });
        setComment('');
    };

    const nextExample = () => {
        if (currentIndex < examples.length - 1) {
            setCurrentIndex(currentIndex + 1);
        }
    };

    const currentExample = examples[currentIndex];

    if (!currentExample) return <Text>No examples to review</Text>;

    return (
        <Box p={5}>
            <Heading size="md" mb={6}>Dataset Review</Heading>
            <Box borderWidth="1px" borderRadius="lg" p={5}>
                <VStack spacing={4} align="stretch">
                    <HStack justify="space-between">
                        <Text fontWeight="bold">Example {currentIndex + 1} of {examples.length}</Text>
                        <Badge colorScheme="blue">{currentExample.status || 'Pending'}</Badge>
                    </HStack>

                    <Box p={4} bg="gray.50" borderRadius="md">
                        <pre style={{ whiteSpace: 'pre-wrap' }}>
                            {JSON.stringify(currentExample.content, null, 2)}
                        </pre>
                    </Box>

                    <Divider />

                    <HStack>
                        <Button colorScheme="green" onClick={handleApprove}>Approve</Button>
                        <Button colorScheme="red" onClick={handleReject}>Reject</Button>
                    </HStack>

                    <Box>
                        <Text mb={2}>Add Comment:</Text>
                        <Textarea
                            value={comment}
                            onChange={(e) => setComment(e.target.value)}
                            placeholder="Type your feedback here..."
                        />
                        <Button mt={2} size="sm" onClick={handleComment}>Post Comment</Button>
                    </Box>
                </VStack>
            </Box>
        </Box>
    );
};

export default ReviewInterface;

