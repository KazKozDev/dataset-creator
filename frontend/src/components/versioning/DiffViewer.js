import React, { useState, useEffect } from 'react';
import { Box, Select, VStack, Heading, Text } from '@chakra-ui/react';
import ReactDiffViewer from 'react-diff-viewer-continued';
import axios from 'axios';

const DiffViewer = ({ datasetId }) => {
    const [versions, setVersions] = useState([]);
    const [leftVersion, setLeftVersion] = useState('');
    const [rightVersion, setRightVersion] = useState('');
    const [leftContent, setLeftContent] = useState('');
    const [rightContent, setRightContent] = useState('');

    useEffect(() => {
        if (datasetId) {
            fetchVersions();
        }
    }, [datasetId]);

    useEffect(() => {
        if (leftVersion) fetchVersionContent(leftVersion, setLeftContent);
    }, [leftVersion]);

    useEffect(() => {
        if (rightVersion) fetchVersionContent(rightVersion, setRightContent);
    }, [rightVersion]);

    const fetchVersions = async () => {
        try {
            // Mock endpoint for versions list
            // const response = await axios.get(`http://localhost:8000/api/datasets/${datasetId}/versions`);
            // setVersions(response.data.versions);

            // Mock data for now
            setVersions([
                { id: 'v1', created_at: '2023-01-01' },
                { id: 'v2', created_at: '2023-01-02' }
            ]);
        } catch (error) {
            console.error(error);
        }
    };

    const fetchVersionContent = async (versionId, setContent) => {
        try {
            // Mock content fetching
            // const response = await axios.get(`http://localhost:8000/api/datasets/${datasetId}/versions/${versionId}/content`);
            // setContent(JSON.stringify(response.data, null, 2));

            if (versionId === 'v1') {
                setContent(JSON.stringify({ messages: [{ role: 'user', content: 'hello' }] }, null, 2));
            } else {
                setContent(JSON.stringify({ messages: [{ role: 'user', content: 'hello world' }] }, null, 2));
            }
        } catch (error) {
            console.error(error);
        }
    };

    return (
        <Box p={5}>
            <Heading size="md" mb={6}>Version History</Heading>
            <VStack spacing={4} align="stretch">
                <Heading size="sm">Version Comparison</Heading>
                <Box display="flex" gap={4}>
                    <Box flex={1}>
                        <Text mb={2}>Original Version</Text>
                        <Select placeholder="Select version" value={leftVersion} onChange={(e) => setLeftVersion(e.target.value)}>
                            {versions.map((v) => (
                                <option key={v.id} value={v.id}>{v.id} ({v.created_at})</option>
                            ))}
                        </Select>
                    </Box>
                    <Box flex={1}>
                        <Text mb={2}>Modified Version</Text>
                        <Select placeholder="Select version" value={rightVersion} onChange={(e) => setRightVersion(e.target.value)}>
                            {versions.map((v) => (
                                <option key={v.id} value={v.id}>{v.id} ({v.created_at})</option>
                            ))}
                        </Select>
                    </Box>
                </Box>

                <Box border="1px solid #e2e2e2" borderRadius="md" overflow="hidden">
                    <ReactDiffViewer
                        oldValue={leftContent}
                        newValue={rightContent}
                        splitView={true}
                        leftTitle={`Version ${leftVersion || '...'}`}
                        rightTitle={`Version ${rightVersion || '...'}`}
                    />
                </Box>
            </VStack>
        </Box>
    );
};

export default DiffViewer;
